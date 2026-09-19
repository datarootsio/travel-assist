-- The participant credential handout: one row per person, in order.
--
-- Included by 003_readonly_role.sql (via \ir) so the password derivation lives
-- in exactly one place, and runnable on its own to reprint the list later:
--
--   psql "<admin url>" -v n_students=19 -v student_secret="…" -f handout.sql
--
-- Nothing stores these. They are derived from STUDENT_SECRET, so reprinting is
-- the recovery path when somebody mislays theirs — no reprovisioning, and no
-- password list sitting in a file. Must stay byte-identical to the derivation
-- in 003_readonly_role.sql; a test in tests/_repo pins that.

SELECT role_name AS pguser,
       left(encode(sha256((:'student_secret' || ':' || role_name)::bytea), 'hex'), 24)
         AS pgpassword
FROM (
    SELECT 'travel_assist_s' || lpad(i::text, 2, '0') AS role_name
    FROM generate_series(1, :n_students) AS i
) r
ORDER BY pguser;
