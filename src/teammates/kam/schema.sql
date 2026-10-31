-- KAM digest tables (Postgres 15+). Idempotent: runs on every start after the support schema.
-- Everything else the digest reads already exists: support_reviewer (the KAM), ticket and
-- audit_log (the support hand-off), merchant_signal (friction), message_log and trace_span.

-- The KAM's calendar (mocked calendar tool in this build; a calendar sync writes here later).
CREATE TABLE IF NOT EXISTS kam_meeting (
    kam_email    text NOT NULL,                  -- support_reviewer.email of the KAM
    id           bigint GENERATED ALWAYS AS IDENTITY,
    starts_at    timestamptz NOT NULL,
    ends_at      timestamptz NOT NULL,
    merchant_id  bigint,                         -- NULL = internal meeting
    title        text NOT NULL,
    location     text,
    PRIMARY KEY (kam_email, id),
    CHECK (ends_at > starts_at)
);
CREATE UNIQUE INDEX IF NOT EXISTS kam_meeting_slot_idx ON kam_meeting (kam_email, starts_at, title);

-- Competitor technology updates from the fixed sources (config.toml [[sources]]).
-- title/body are UNTRUSTED third-party text: InputGuard + wrap_untrusted before any LLM.
CREATE TABLE IF NOT EXISTS competitor_update (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source        text NOT NULL,
    competitor    text NOT NULL,
    title         text NOT NULL,
    body          text NOT NULL,
    url           text NOT NULL UNIQUE CHECK (url LIKE 'https://%'),
    published_at  timestamptz NOT NULL,
    fetched_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS competitor_update_published_idx ON competitor_update (published_at);
