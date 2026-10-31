-- Support desk schema (Postgres 15+). Idempotent: runs on every start, before the FSE schema
-- whose trace_span table it shares.
--
-- SHARDING (designed, not implemented). Shard key = merchant_id. Every merchant-owned table
-- carries it and its primary key starts with it, so one merchant's tickets, audit trail and
-- friction signal sit on one shard and every review action is a single-shard transaction;
-- routes carry it too (/support/tickets/<merchant_id>/<id>). With Citus:
--   SELECT create_reference_table('support_reviewer');
--   SELECT create_distributed_table('ticket', 'merchant_id');
--   SELECT create_distributed_table(t, 'merchant_id', colocate_with => 'ticket')
--     FOR audit_log, merchant_signal;
-- The reviewer queue (status + assignee) fans out across shards; it is small and indexed.
--
-- CACHE. Redis keys support:{merchant_id}:{ticket_id}:{profile|security|txns}; deleted by the
-- app on every ticket state change, short TTLs on the sensitive parts (config.toml [cache]).

CREATE TABLE IF NOT EXISTS support_reviewer (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name           text NOT NULL,
    email          text NOT NULL UNIQUE,            -- login; KAM tickets are assigned to it
    password_hash  text NOT NULL,                   -- scrypt$salt$hash
    role           text NOT NULL CHECK (role IN ('kam', 'support_agent')),
    active         boolean NOT NULL DEFAULT true
);

-- The central tickets table: everything about a ticket, from intake to the reply sent.
CREATE TABLE IF NOT EXISTS ticket (
    merchant_id     bigint NOT NULL,                -- 0 = sender not a known merchant contact
    id              bigint GENERATED ALWAYS AS IDENTITY,
    channel         text NOT NULL CHECK (channel IN ('email', 'whatsapp', 'call')),
    contact         text NOT NULL,                  -- reply goes here only if verified
    text            text NOT NULL,                  -- UNTRUSTED customer text, after InputGuard
    amount          double precision,               -- as the customer stated it
    received_at     timestamptz NOT NULL,           -- first message
    last_contact_at timestamptz NOT NULL,           -- latest message (repeat-contact window)
    contacts        int NOT NULL DEFAULT 1,         -- repeat contacts on this ticket
    status          text NOT NULL DEFAULT 'new'
                    CHECK (status IN ('new', 'pending_review', 'resolved', 'manual')),
    type            text CHECK (type IN ('faq', 'status', 'refund', 'dispute')),
    confidence      real,
    guard_flags     text[] NOT NULL DEFAULT '{}',
    faq_id          text,
    score           real,
    band            text,
    risk_breakdown  jsonb,                          -- factors, ratings, overrides, reasons
    route           text CHECK (route IN ('auto', 'review')),
    assigned_role   text CHECK (assigned_role IN ('kam', 'support_agent')),
    assigned_to     text,                           -- reviewer email; NULL = any of the role
    merchant_name   text,
    tier            text,
    language        text,
    draft           text,                           -- the AI's reply
    final_reply     text,                           -- what was actually sent
    reject_reason   text,
    trace_id        text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (merchant_id, id)
);
CREATE INDEX IF NOT EXISTS ticket_merchant_status_idx ON ticket (merchant_id, status);
CREATE INDEX IF NOT EXISTS ticket_status_idx ON ticket (status, assigned_role, assigned_to);

-- Every action on a ticket, append-only; review actions carry the edit diff and risk band.
-- ponytail: not partitioned; partition by month like application_event when it grows.
CREATE TABLE IF NOT EXISTS audit_log (
    merchant_id  bigint NOT NULL,
    id           bigint GENERATED ALWAYS AS IDENTITY,
    at           timestamptz NOT NULL DEFAULT now(),
    ticket_id    bigint NOT NULL,
    actor        text NOT NULL,                     -- 'ai' or the reviewer's email
    action       text NOT NULL,
    band         text,
    payload      jsonb NOT NULL DEFAULT '{}',
    trace_id     text,
    PRIMARY KEY (merchant_id, id)
);
CREATE INDEX IF NOT EXISTS audit_log_ticket_idx ON audit_log (merchant_id, ticket_id, at);
CREATE INDEX IF NOT EXISTS audit_log_action_idx ON audit_log (action, at);

-- Friction signal on the merchant record, read by the KAM and collections teammates.
CREATE TABLE IF NOT EXISTS merchant_signal (
    merchant_id     bigint PRIMARY KEY,
    friction        int NOT NULL DEFAULT 0,         -- +1 per contact, +1 if a human was needed
    tickets         int NOT NULL DEFAULT 0,
    last_ticket_id  bigint,
    last_outcome    text,
    updated_at      timestamptz NOT NULL DEFAULT now()
);
