-- Supabase RPC 함수 대체 SQL

-- update_crawling_stats 함수 생성
CREATE OR REPLACE FUNCTION update_crawling_stats(
    p_site_id VARCHAR,
    p_success BOOLEAN,
    p_details JSONB DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    IF p_success THEN
        UPDATE crawling_sites
        SET 
            crawl_count = COALESCE(crawl_count, 0) + 1,
            success_count = COALESCE(success_count, 0) + 1,
            last_crawl_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE site_id = p_site_id;
    ELSE
        UPDATE crawling_sites
        SET 
            crawl_count = COALESCE(crawl_count, 0) + 1,
            last_crawl_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE site_id = p_site_id;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- 권한 부여
GRANT EXECUTE ON FUNCTION update_crawling_stats TO aimonitor;