-- Collections manager schema (Postgres 15+). Idempotent: runs on every start, after the FSE
-- schema, whose job_run, trace_span and message_log tables it shares. Amounts: whole rupees.
--
-- SHARDING. Shard key = borrower_id. Every borrower-owned table carries it, and its primary key
-- and foreign keys start with it, so one borrower's loans, instalments, applications and audit
-- trail sit on one shard: every application check and decision is a single-shard transaction.
-- With Citus:
--   SELECT create_reference_table(t) FOR area, area_target, collection_manager, manager_area;
--   SELECT create_distributed_table('borrower', 'id');
--   SELECT create_distributed_table(t, 'borrower_id', colocate_with => 'borrower')
--     FOR loan, instalment, loan_application, application_event;
-- The brief's per-area reads fan out across shards (once a day, fine). Append-only
-- application_event is also partitioned by month (LoanStore.ensure_partitions).
--
-- REDIS RECONCILIATION. Triggers bump area.version whenever anything the brief shows changes
-- (borrower, loan, instalment, area_target), whoever wrote it: the app, a payment system, or SQL
-- typed into the Neon console. Cached brief data is stamped with the version it was built from;
-- a reader compares the stamp with area.version (a primary-key read) and re-reads Postgres on a
-- mismatch. Redis can never serve stale numbers, and a lost cache DEL changes nothing.

-- One-time: replace the first POC tables (no shard key in their keys). Demo data only.
DO $$ BEGIN
    IF to_regclass('borrower') IS NOT NULL AND to_regclass('area') IS NULL THEN
        DROP TABLE IF EXISTS loan_application, instalment, loan, area_target, borrower CASCADE;
    END IF;
END $$;

-- ===== Reference tables (small, replicated to every shard) ================================

CREATE TABLE IF NOT EXISTS area (
    name     text PRIMARY KEY,
    version  bigint NOT NULL DEFAULT 0                -- bumped by triggers, stamps cached data
);

CREATE TABLE IF NOT EXISTS area_target (
    area    text NOT NULL REFERENCES area(name),
    month   date NOT NULL CHECK (extract(day FROM month) = 1),
    target  bigint NOT NULL CHECK (target >= 0),
    PRIMARY KEY (area, month)
);

-- RBAC. collection_manager: sees and decides applications in their areas, gets the brief.
-- area_collection_manager: sees them, cannot decide.
CREATE TABLE IF NOT EXISTS collection_manager (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name            text NOT NULL,
    email           text NOT NULL UNIQUE,             -- login, and the brief's only recipient
    password_hash   text NOT NULL,                    -- scrypt$salt$hash
    role            text NOT NULL CHECK (role IN ('collection_manager', 'area_collection_manager')),
    email_verified  boolean NOT NULL DEFAULT false,   -- output guard: brief only to verified
    active          boolean NOT NULL DEFAULT true
);

CREATE TABLE IF NOT EXISTS manager_area (
    manager_id  bigint NOT NULL REFERENCES collection_manager(id),
    area        text NOT NULL REFERENCES area(name),
    PRIMARY KEY (manager_id, area)
);
CREATE INDEX IF NOT EXISTS manager_area_area_idx ON manager_area (area);

-- ===== Borrower-owned tables (distributed by borrower_id) ==================================

-- A customer: existing borrower or applicant. Bureau, background and financial fields are what
-- the application tools read; in production they're filled from those systems.
CREATE TABLE IF NOT EXISTS borrower (
    id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,   -- = borrower_id
    name              text NOT NULL,
    business          text NOT NULL,
    phone             text NOT NULL UNIQUE,           -- Citus: becomes a phone -> id lookup table
    area              text NOT NULL REFERENCES area(name),
    credit_score      smallint CHECK (credit_score BETWEEN 300 AND 900),
    background_flags  text[] NOT NULL DEFAULT '{}',   -- e.g. {fraud_case, wilful_default}
    monthly_revenue   bigint,
    monthly_expenses  bigint
);
CREATE INDEX IF NOT EXISTS borrower_area_idx ON borrower (area);
CREATE INDEX IF NOT EXISTS borrower_name_idx ON borrower (name);  -- output guard name lookup

CREATE TABLE IF NOT EXISTS loan (
    borrower_id   bigint NOT NULL REFERENCES borrower(id),
    id            bigint GENERATED ALWAYS AS IDENTITY,
    principal     bigint NOT NULL CHECK (principal > 0),
    emi           bigint NOT NULL CHECK (emi > 0),
    disbursed_on  date NOT NULL,
    status        text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'closed')),
    PRIMARY KEY (borrower_id, id)
);

-- Created up front for the whole tenure, so not time-partitioned (future months would need
-- partitions years ahead); the shard key is what spreads it.
CREATE TABLE IF NOT EXISTS instalment (
    borrower_id  bigint NOT NULL,
    id           bigint GENERATED ALWAYS AS IDENTITY,
    loan_id      bigint NOT NULL,
    due_on       date NOT NULL,
    amount       bigint NOT NULL CHECK (amount > 0),
    paid_on      date,
    PRIMARY KEY (borrower_id, id),
    UNIQUE (borrower_id, loan_id, due_on),            -- also the per-loan read index
    FOREIGN KEY (borrower_id, loan_id) REFERENCES loan (borrower_id, id)
);
-- Pending collections: only unpaid rows. Collected this month: only paid rows, by date.
CREATE INDEX IF NOT EXISTS instalment_unpaid_idx ON instalment (due_on, borrower_id)
    WHERE paid_on IS NULL;
CREATE INDEX IF NOT EXISTS instalment_paid_idx ON instalment (paid_on, borrower_id)
    WHERE paid_on IS NOT NULL;

-- One application. The AI fills checks, summary and recommendation; only a collection_manager
-- sets approved/rejected (RBAC enforced in the UPDATE itself, see LoanStore.decide).
CREATE TABLE IF NOT EXISTS loan_application (
    borrower_id      bigint NOT NULL REFERENCES borrower(id),
    id               bigint GENERATED ALWAYS AS IDENTITY,
    amount           bigint NOT NULL CHECK (amount > 0),
    tenure_months    smallint NOT NULL CHECK (tenure_months BETWEEN 1 AND 120),
    purpose          text NOT NULL,                   -- after the input guard
    checks           jsonb NOT NULL,                  -- {background: {ok, detail}, ...}
    summary          text NOT NULL,
    recommendation   text NOT NULL CHECK (recommendation IN ('can', 'cannot')),
    overridden       boolean NOT NULL DEFAULT false,  -- LLM said "can" while a check failed
    status           text NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'approved', 'rejected')),
    decided_by       bigint REFERENCES collection_manager(id),
    decided_at       timestamptz,
    note             text,
    trace_id         text,
    created_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (borrower_id, id),
    CHECK ((status = 'pending') = (decided_by IS NULL))
);
CREATE INDEX IF NOT EXISTS loan_application_pending_idx ON loan_application (created_at)
    WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS loan_application_decided_idx ON loan_application (decided_at)
    WHERE status <> 'pending';

-- Audit log: append-only, written in the same transaction as the change it records.
CREATE TABLE IF NOT EXISTS application_event (
    borrower_id     bigint NOT NULL,
    at              timestamptz NOT NULL DEFAULT now(),
    id              bigint GENERATED ALWAYS AS IDENTITY,
    application_id  bigint NOT NULL,
    type            text NOT NULL CHECK (type IN ('submitted', 'assessed', 'approved',
                                                  'rejected')),
    actor           text NOT NULL,                    -- 'ai' or the manager's email
    payload         jsonb NOT NULL DEFAULT '{}',
    trace_id        text,
    PRIMARY KEY (borrower_id, at, id)
) PARTITION BY RANGE (at);
CREATE INDEX IF NOT EXISTS application_event_app_idx
    ON application_event (borrower_id, application_id, at);

-- ===== Cache version triggers ===============================================================

CREATE OR REPLACE FUNCTION coll_bump_area_version() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    r record;
BEGIN
    IF TG_OP = 'DELETE' THEN r := OLD; ELSE r := NEW; END IF;
    IF TG_TABLE_NAME = 'borrower' AND TG_OP = 'UPDATE' THEN  -- may move between areas
        UPDATE area SET version = version + 1 WHERE name IN (OLD.area, NEW.area);
    ELSIF TG_TABLE_NAME = 'borrower' THEN
        UPDATE area SET version = version + 1 WHERE name = r.area;
    ELSIF TG_TABLE_NAME = 'area_target' THEN
        UPDATE area SET version = version + 1 WHERE name = r.area;
    ELSE  -- loan, instalment
        UPDATE area SET version = version + 1
        WHERE name = (SELECT area FROM borrower WHERE id = r.borrower_id);
    END IF;
    RETURN NULL;
END $$;
-- ponytail: one version row per area serialises concurrent payment writes in that area;
-- move to statement-level triggers or CDC if payment volume makes it hot.
CREATE OR REPLACE TRIGGER borrower_version AFTER INSERT OR UPDATE OR DELETE ON borrower
    FOR EACH ROW EXECUTE FUNCTION coll_bump_area_version();
CREATE OR REPLACE TRIGGER loan_version AFTER INSERT OR UPDATE OR DELETE ON loan
    FOR EACH ROW EXECUTE FUNCTION coll_bump_area_version();
CREATE OR REPLACE TRIGGER instalment_version AFTER INSERT OR UPDATE OR DELETE ON instalment
    FOR EACH ROW EXECUTE FUNCTION coll_bump_area_version();
CREATE OR REPLACE TRIGGER area_target_version AFTER INSERT OR UPDATE OR DELETE ON area_target
    FOR EACH ROW EXECUTE FUNCTION coll_bump_area_version();
