-- FSE copilot schema (Postgres 15+). Idempotent: safe to run on every start.
-- Every table carries fse_id so it can be sharded by fse_id later (e.g. Citus) without redesign.
-- Monthly partitions are created by store.ensure_partitions(); old months are dropped, not deleted.

CREATE TABLE IF NOT EXISTS team_lead (
    id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name   text NOT NULL,
    email  text NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS fse (
    id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name     text NOT NULL,
    phone    text NOT NULL UNIQUE,              -- digits only; inbound lookup key
    area     text NOT NULL,
    lead_id  bigint REFERENCES team_lead(id)
);
CREATE INDEX IF NOT EXISTS fse_lead_idx ON fse (lead_id);

CREATE TABLE IF NOT EXISTS merchant (
    id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name   text NOT NULL,
    area   text NOT NULL,
    UNIQUE (area, name)
);

CREATE TABLE IF NOT EXISTS reason (
    code   text PRIMARY KEY,
    label  text NOT NULL
);

-- Source of truth for the day's work. One row per FSE x merchant x day.
CREATE TABLE IF NOT EXISTS task (
    id           bigint GENERATED ALWAYS AS IDENTITY,
    day          date NOT NULL,
    fse_id       bigint NOT NULL REFERENCES fse(id),
    merchant_id  bigint NOT NULL REFERENCES merchant(id),
    kind         text NOT NULL,
    priority     smallint NOT NULL,             -- 1 = highest
    effort       smallint NOT NULL,             -- 1..3, higher = longer
    status       text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done', 'not_done')),
    reason_code  text REFERENCES reason(code),
    reason_text  text,
    done_at      timestamptz,
    updated_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (day, id),
    -- Also the read index: "this FSE's tasks today" hits one partition, one index range.
    UNIQUE (fse_id, day, merchant_id)
) PARTITION BY RANGE (day);

-- Append-only audit trail of every change, written in the same transaction as the change.
CREATE TABLE IF NOT EXISTS task_event (
    id       bigint GENERATED ALWAYS AS IDENTITY,
    at       timestamptz NOT NULL DEFAULT now(),
    fse_id   bigint NOT NULL,
    task_id  bigint,
    type     text NOT NULL CHECK (type IN ('done', 'reason', 'feedback')),
    payload  jsonb NOT NULL DEFAULT '{}',
    PRIMARY KEY (at, id)
) PARTITION BY RANGE (at);
CREATE INDEX IF NOT EXISTS task_event_fse_idx ON task_event (fse_id, at);

CREATE TABLE IF NOT EXISTS feedback (
    id           bigint GENERATED ALWAYS AS IDENTITY,
    at           timestamptz NOT NULL DEFAULT now(),
    fse_id       bigint NOT NULL,
    merchant_id  bigint,
    body         text,
    audio_url    text,
    PRIMARY KEY (at, id)
) PARTITION BY RANGE (at);
CREATE INDEX IF NOT EXISTS feedback_fse_idx ON feedback (fse_id, at);

-- Small read model: one row per FSE per day, kept in sync inside every task write.
-- Team-lead views read this instead of scanning task.
CREATE TABLE IF NOT EXISTS fse_day (
    fse_id      bigint NOT NULL,
    day         date NOT NULL,
    total       int NOT NULL DEFAULT 0,
    done        int NOT NULL DEFAULT 0,
    not_done    int NOT NULL DEFAULT 0,
    reasons     jsonb NOT NULL DEFAULT '{}',     -- {"closed": 1, "other": 2}
    updated_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (fse_id, day)
);
CREATE INDEX IF NOT EXISTS fse_day_day_idx ON fse_day (day);
