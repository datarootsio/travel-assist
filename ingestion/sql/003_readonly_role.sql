-- One read-only role per participant.
--
-- Requires three psql variables, e.g.
--   psql -v n_students=19 -v student_secret="'…'" -v max_connections=3 \
--        -f 003_readonly_role.sql
--
-- ONE ROLE PER PARTICIPANT, and the reason is the connection cap. An earlier
-- version of this file created a single shared `travel_assist_student` and set
-- `CONNECTION LIMIT` on it — but Postgres applies that limit per *role*, not
-- per person, so the cap was shared by the whole room. With a limit of 3, the
-- fourth participant to open a pool was refused. That is exactly the failure
-- the limit exists to prevent, and it would have arrived mid-exercise, to
-- everyone at once, with no obvious cause.
--
-- Two limits still matter more than the permissions do. One participant writing
-- an accidental cross join, or opening a connection per notebook cell, must not
-- be able to starve the other eighteen. Now they cannot.

\set ON_ERROR_STOP on

-- Handed to the block below as GUCs: psql does not interpolate `:vars` inside a
-- dollar-quoted body, so they cannot be referenced there directly.
SET travel_assist.n_students      = :'n_students';
SET travel_assist.max_connections = :'max_connections';
SET travel_assist.secret          = :'student_secret';

DO $$
DECLARE
    n         int  := current_setting('travel_assist.n_students')::int;
    max_conn  int  := current_setting('travel_assist.max_connections')::int;
    secret    text := current_setting('travel_assist.secret');
    role_name text;
    pw        text;
BEGIN
    IF n < 1 OR n > 99 THEN
        RAISE EXCEPTION 'n_students must be between 1 and 99, got %', n;
    END IF;

    FOR i IN 1..n LOOP
        role_name := 'travel_assist_s' || lpad(i::text, 2, '0');

        -- Distinct per participant, and derived rather than stored: the same
        -- secret regenerates any one password later, so somebody who mislays
        -- theirs on the morning does not cost the room a reprovision.
        --
        -- Distinct passwords are what make the credential gate hold. Students
        -- receive these only after the architecture drawing; with one shared
        -- password the first person through the gate can open it for everyone.
        --
        -- sha256() and encode() are core (PG 11+), so no extension is needed.
        pw := left(encode(sha256((secret || ':' || role_name)::bytea), 'hex'), 24);

        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = role_name) THEN
            EXECUTE format('CREATE ROLE %I LOGIN', role_name);
        END IF;

        EXECUTE format('ALTER ROLE %I PASSWORD %L', role_name, pw);

        -- A runaway query gives up instead of holding a connection all afternoon.
        EXECUTE format('ALTER ROLE %I SET statement_timeout = %L', role_name, '30s');

        -- Per person now, not per room.
        EXECUTE format('ALTER ROLE %I CONNECTION LIMIT %s', role_name, max_conn);

        EXECUTE format('GRANT CONNECT ON DATABASE %I TO %I',
                       current_database(), role_name);
        EXECUTE format('GRANT USAGE ON SCHEMA public TO %I', role_name);

        -- SELECT on exactly the three tables, and nothing else. No INSERT, no
        -- UPDATE, no DELETE, no sequence access: students must not be able to
        -- break each other.
        EXECUTE format('GRANT SELECT ON documents, chunks, index_metadata TO %I',
                       role_name);

        -- Future tables are not granted by default; adding one is a deliberate act.
        EXECUTE format(
            'ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON TABLES FROM %I',
            role_name);
    END LOOP;
END
$$;

-- The handout: one row per participant. Kept in a sibling file so the password
-- derivation above is not duplicated, and so it can be reprinted on its own.
\ir handout.sql
