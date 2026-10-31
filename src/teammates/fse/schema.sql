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

-- Location: FSE base and merchant shop. Plain lat/lng; distance is only used to order allotment.
ALTER TABLE fse      ADD COLUMN IF NOT EXISTS address text;
ALTER TABLE fse      ADD COLUMN IF NOT EXISTS lat double precision;
ALTER TABLE fse      ADD COLUMN IF NOT EXISTS lng double precision;
ALTER TABLE merchant ADD COLUMN IF NOT EXISTS address text;
ALTER TABLE merchant ADD COLUMN IF NOT EXISTS lat double precision;
ALTER TABLE merchant ADD COLUMN IF NOT EXISTS lng double precision;

-- Open work per area. The morning plan allots open projects in the FSE's area into task rows.
-- open -> allotted (task created) -> closed (task done); a not-done task puts it back to open.
CREATE TABLE IF NOT EXISTS project (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    area         text NOT NULL,
    merchant_id  bigint NOT NULL REFERENCES merchant(id),
    kind         text NOT NULL,
    priority     smallint NOT NULL,             -- 1 = highest
    effort       smallint NOT NULL,             -- 1..3, higher = longer
    status       text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'allotted', 'closed')),
    fse_id       bigint REFERENCES fse(id),     -- set while allotted
    allotted_on  date,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now()
);
-- Allotment read: "open projects in this area, best first". Partial, so it only holds open rows.
CREATE INDEX IF NOT EXISTS project_open_area_idx ON project (area, priority) WHERE status = 'open';
-- No duplicate live project for the same merchant and job.
CREATE UNIQUE INDEX IF NOT EXISTS project_live_uniq ON project (merchant_id, kind)
    WHERE status <> 'closed';

ALTER TABLE task ADD COLUMN IF NOT EXISTS project_id bigint;

-- Customer review per done task. The bot creates the row (body NULL) when it asks
-- "what did <merchant> say?"; the FSE's text or voice reply fills it. One review per task visit.
-- RBAC: rows are only created for the FSE's own done tasks and only filled by that fse_id.
CREATE TABLE IF NOT EXISTS review (
    id           bigint GENERATED ALWAYS AS IDENTITY,
    day          date NOT NULL,
    task_id      bigint NOT NULL,
    fse_id       bigint NOT NULL REFERENCES fse(id),
    merchant_id  bigint NOT NULL REFERENCES merchant(id),
    body         text,
    audio_url    text,
    asked_at     timestamptz NOT NULL DEFAULT now(),
    answered_at  timestamptz,
    PRIMARY KEY (day, id),
    UNIQUE (day, task_id)
) PARTITION BY RANGE (day);
-- "Reviews for this merchant", newest first.
CREATE INDEX IF NOT EXISTS review_merchant_idx ON review (merchant_id, day);
-- The waiting question for an FSE: tiny partial index, only unanswered rows.
CREATE INDEX IF NOT EXISTS review_pending_idx ON review (fse_id, day) WHERE answered_at IS NULL;

-- Scheduled jobs: one row per job per day, claimed before the job runs, so a restart, a deploy
-- or a second instance never sends the same job twice.
CREATE TABLE IF NOT EXISTS job_run (
    job         text NOT NULL,
    day         date NOT NULL,
    started_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (job, day)
);

-- ===== Observability ========================================================================
-- Trace spans (see tracing.py). One trace = one job run, one FSE in a job, or one inbound msg.
CREATE TABLE IF NOT EXISTS trace_span (
    started_at   timestamptz NOT NULL,
    id           text NOT NULL,
    trace_id     text NOT NULL,
    parent_id    text,
    name         text NOT NULL,
    path         text NOT NULL,                 -- "job.morning > fse > plan"
    fse_id       bigint,
    status       text NOT NULL CHECK (status IN ('ok', 'error')),
    error        text,                          -- "KeyError: 'x' at store.py:212 in plan"
    duration_ms  real NOT NULL,
    attrs        jsonb NOT NULL DEFAULT '{}',
    PRIMARY KEY (started_at, id)
) PARTITION BY RANGE (started_at);
CREATE INDEX IF NOT EXISTS trace_span_trace_idx ON trace_span (trace_id);
CREATE INDEX IF NOT EXISTS trace_span_error_idx ON trace_span (started_at) WHERE status = 'error';
CREATE INDEX IF NOT EXISTS trace_span_root_idx ON trace_span (name, started_at)
    WHERE parent_id IS NULL;

-- Every WhatsApp message in and out. Vonage delivery callbacks fill the delivery_* columns.
CREATE TABLE IF NOT EXISTS message_log (
    id              bigint GENERATED ALWAYS AS IDENTITY,
    at              timestamptz NOT NULL DEFAULT now(),
    direction       text NOT NULL CHECK (direction IN ('out', 'in')),
    fse_id          bigint,
    phone_masked    text NOT NULL,
    kind            text NOT NULL,              -- text / list / reply / audio / location
    body            text,
    trace_id        text,
    ok              boolean NOT NULL,           -- out: Vonage accepted it; in: registered FSE
    http_status     int,
    error           text,
    message_uuid    text,                       -- Vonage id, joins delivery callbacks
    delivery        text,                       -- delivered / read / rejected / undeliverable
    delivery_error  text,
    delivery_at     timestamptz,
    PRIMARY KEY (at, id)
) PARTITION BY RANGE (at);
CREATE INDEX IF NOT EXISTS message_log_uuid_idx ON message_log (message_uuid)
    WHERE message_uuid IS NOT NULL;
CREATE INDEX IF NOT EXISTS message_log_fse_idx ON message_log (fse_id, at);

-- Views are recreated on every start so a column change never blocks the migration.
DROP VIEW IF EXISTS v_daily_metrics, v_job_runs, v_root_causes;

-- Root cause per failed trace: the deepest failed step (no failed child), newest first.
CREATE VIEW v_root_causes AS
SELECT (s.started_at AT TIME ZONE 'Asia/Kolkata') AS at_ist, s.path AS failed_step, s.error,
       s.fse_id, s.attrs, s.trace_id
FROM trace_span s
WHERE s.status = 'error'
  AND NOT EXISTS (SELECT 1 FROM trace_span c
                  WHERE c.trace_id = s.trace_id AND c.parent_id = s.id AND c.status = 'error')
ORDER BY s.started_at DESC;

-- Each scheduled job run: when it started, how late, how long, how many steps failed.
CREATE VIEW v_job_runs AS
SELECT (s.started_at AT TIME ZONE 'Asia/Kolkata')::date AS day, s.name AS job,
       (s.started_at AT TIME ZONE 'Asia/Kolkata') AS started_ist,
       (s.attrs ->> 'minutes_late')::int AS minutes_late, (s.attrs ->> 'fses')::int AS fses,
       round(s.duration_ms) AS duration_ms, s.status,
       (SELECT count(*) FROM trace_span e
        WHERE e.trace_id = s.trace_id AND e.name = 'fse' AND e.status = 'error') AS failed_fses,
       s.trace_id
FROM trace_span s
WHERE s.parent_id IS NULL AND s.name LIKE 'job.%'
ORDER BY s.started_at DESC;

-- Daily business + delivery + reliability numbers (IST days).
CREATE VIEW v_daily_metrics AS
WITH d AS (
    SELECT day, count(*) AS fses_planned, sum(total) AS tasks, sum(done) AS done,
           sum(not_done) AS not_done
    FROM fse_day GROUP BY day),
m AS (
    SELECT (at AT TIME ZONE 'Asia/Kolkata')::date AS day,
           count(*) FILTER (WHERE direction = 'out') AS sent,
           count(*) FILTER (WHERE direction = 'out' AND NOT ok) AS send_failed,
           count(*) FILTER (WHERE direction = 'out' AND delivery IN ('delivered', 'read'))
               AS delivered,
           count(*) FILTER (WHERE direction = 'out'
                            AND delivery IN ('rejected', 'undeliverable', 'failed'))
               AS undelivered,
           count(*) FILTER (WHERE direction = 'in') AS received,
           count(*) FILTER (WHERE direction = 'in' AND NOT ok) AS rejected_senders
    FROM message_log GROUP BY 1),
r AS (SELECT day, count(answered_at) AS reviews FROM review GROUP BY day),
rc AS (SELECT at_ist::date AS day, count(*) AS root_causes FROM v_root_causes GROUP BY 1),
s AS (
    SELECT (started_at AT TIME ZONE 'Asia/Kolkata')::date AS day,
           round((percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms)
                  FILTER (WHERE name = 'inbound'))::numeric) AS inbound_p50_ms,
           round((percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms)
                  FILTER (WHERE name = 'inbound'))::numeric) AS inbound_p95_ms
    FROM trace_span GROUP BY 1)
SELECT day, fses_planned, tasks, done, not_done,
       round(100.0 * done / nullif(tasks, 0), 1) AS done_pct,
       reviews, round(100.0 * reviews / nullif(done, 0), 1) AS review_coverage_pct,
       sent, send_failed, delivered, undelivered, received, rejected_senders,
       root_causes, inbound_p50_ms, inbound_p95_ms
FROM d FULL JOIN m USING (day) FULL JOIN r USING (day) FULL JOIN rc USING (day)
     FULL JOIN s USING (day)
ORDER BY day DESC;
