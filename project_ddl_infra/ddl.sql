-- Catalogs and schemas (mlops_{env}.jayanthk) are created by the course admin.
-- Only mlops_dev.jayanthk is accessible for this student; acc/prd schemas
-- are managed by the course admin.
-- Run this to create your volume inside the dev schema.

create volume if not exists mlops_dev.jayanthk.hf_docs_files;
