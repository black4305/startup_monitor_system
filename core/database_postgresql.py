"""
PostgreSQL 직접 연결 버전의 DatabaseManager
Supabase 대신 로컬/원격 PostgreSQL 서버에 직접 연결
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """PostgreSQL 직접 연결을 위한 데이터베이스 매니저"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # PostgreSQL 연결 설정
        self.database_url = os.getenv('DATABASE_URL', 
            'postgresql://dbuser:password@localhost:5432/startup_monitor')
        
        # 연결 풀 생성 (동시 연결 관리)
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                1,  # 최소 연결 수
                10,  # 최대 연결 수
                self.database_url
            )
            logger.info("✅ PostgreSQL 연결 풀 생성 성공")
            
            # 테이블 초기화
            self._initialize_tables()
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL 연결 실패: {str(e)}")
            raise
            
        self._initialized = True
    
    @contextmanager
    def get_connection(self):
        """연결 풀에서 연결을 가져오고 반환하는 컨텍스트 매니저"""
        conn = self.connection_pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.connection_pool.putconn(conn)
    
    def _initialize_tables(self):
        """필요한 테이블이 없으면 생성"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # support_programs 테이블
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS support_programs (
                        id SERIAL PRIMARY KEY,
                        external_id VARCHAR(255) UNIQUE NOT NULL,
                        title TEXT NOT NULL,
                        brief_content TEXT,
                        url TEXT,
                        source_site VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ai_score FLOAT DEFAULT 0,
                        is_active BOOLEAN DEFAULT true,
                        keywords TEXT[] DEFAULT '{}'::TEXT[],
                        deadline DATE,
                        support_amount VARCHAR(255),
                        target_audience TEXT
                    )
                """)
                
                # user_feedback 테이블
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_feedback (
                        id SERIAL PRIMARY KEY,
                        program_id INTEGER REFERENCES support_programs(id),
                        action VARCHAR(50) NOT NULL,
                        reason TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_auto_deleted BOOLEAN DEFAULT false
                    )
                """)
                
                # ai_learning_stats 테이블
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ai_learning_stats (
                        id SERIAL PRIMARY KEY,
                        total_feedback INTEGER DEFAULT 0,
                        positive_feedback INTEGER DEFAULT 0,
                        negative_feedback INTEGER DEFAULT 0,
                        last_training_date TIMESTAMP,
                        model_accuracy FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # crawling_sites 테이블
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS crawling_sites (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        url TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT true,
                        selector_config JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 인덱스 생성
                cur.execute("CREATE INDEX IF NOT EXISTS idx_programs_external_id ON support_programs(external_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_programs_ai_score ON support_programs(ai_score DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_programs_is_active ON support_programs(is_active)")
                
                logger.info("✅ 데이터베이스 테이블 초기화 완료")
    
    def get_programs(self, limit: int = 100, offset: int = 0, 
                    active_only: bool = True, order_by: str = 'ai_score DESC') -> List[Dict]:
        """프로그램 목록 조회"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT * FROM support_programs
                    WHERE 1=1
                """
                params = []
                
                if active_only:
                    query += " AND is_active = true"
                
                query += f" ORDER BY {order_by} LIMIT %s OFFSET %s"
                params.extend([limit, offset])
                
                cur.execute(query, params)
                return cur.fetchall()
    
    def get_program_by_id(self, program_id: int) -> Optional[Dict]:
        """ID로 프로그램 조회"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM support_programs WHERE id = %s",
                    (program_id,)
                )
                return cur.fetchone()
    
    def insert_program(self, program_data: Dict) -> Optional[Dict]:
        """새 프로그램 추가"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # 중복 체크
                cur.execute(
                    "SELECT id FROM support_programs WHERE external_id = %s",
                    (program_data['external_id'],)
                )
                if cur.fetchone():
                    logger.info(f"프로그램 이미 존재: {program_data['external_id']}")
                    return None
                
                # 삽입
                cur.execute("""
                    INSERT INTO support_programs 
                    (external_id, title, brief_content, url, source_site, 
                     ai_score, keywords, deadline, support_amount, target_audience)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                """, (
                    program_data['external_id'],
                    program_data['title'],
                    program_data.get('brief_content', ''),
                    program_data.get('url', ''),
                    program_data.get('source_site', ''),
                    program_data.get('ai_score', 0),
                    program_data.get('keywords', []),
                    program_data.get('deadline'),
                    program_data.get('support_amount'),
                    program_data.get('target_audience')
                ))
                return cur.fetchone()
    
    def update_program(self, program_id: int, updates: Dict) -> Optional[Dict]:
        """프로그램 정보 업데이트"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # 동적 쿼리 생성
                set_clauses = []
                params = []
                
                for key, value in updates.items():
                    if key not in ['id', 'created_at']:  # 보호된 필드
                        set_clauses.append(f"{key} = %s")
                        params.append(value)
                
                if not set_clauses:
                    return None
                
                params.append(program_id)
                query = f"""
                    UPDATE support_programs 
                    SET {', '.join(set_clauses)}
                    WHERE id = %s
                    RETURNING *
                """
                
                cur.execute(query, params)
                return cur.fetchone()
    
    def delete_program(self, program_id: int) -> bool:
        """프로그램 삭제 (실제로는 is_active를 false로)"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE support_programs SET is_active = false WHERE id = %s",
                    (program_id,)
                )
                return cur.rowcount > 0
    
    def check_program_exists(self, external_id: str) -> Tuple[bool, bool]:
        """프로그램 존재 여부 및 활성 상태 확인"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT is_active FROM support_programs WHERE external_id = %s",
                    (external_id,)
                )
                result = cur.fetchone()
                
                if result is None:
                    return False, False
                return True, result[0]
    
    def add_feedback(self, program_id: int, action: str, reason: str = None, 
                    is_auto_deleted: bool = False) -> Optional[Dict]:
        """사용자 피드백 추가"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO user_feedback 
                    (program_id, action, reason, is_auto_deleted)
                    VALUES (%s, %s, %s, %s)
                    RETURNING *
                """, (program_id, action, reason, is_auto_deleted))
                return cur.fetchone()
    
    def get_feedback_stats(self) -> Dict:
        """피드백 통계 조회"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN action = 'keep' THEN 1 ELSE 0 END) as keep_count,
                        SUM(CASE WHEN action = 'delete' THEN 1 ELSE 0 END) as delete_count
                    FROM user_feedback
                """)
                result = cur.fetchone()
                return {
                    'total': result[0] or 0,
                    'keep': result[1] or 0,
                    'delete': result[2] or 0
                }
    
    def get_crawling_sites(self, active_only: bool = True) -> List[Dict]:
        """크롤링 사이트 목록 조회"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = "SELECT * FROM crawling_sites"
                if active_only:
                    query += " WHERE is_active = true"
                cur.execute(query)
                return cur.fetchall()
    
    def get_dashboard_stats(self) -> Dict:
        """대시보드 통계 조회"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # 전체 통계
                cur.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN is_active = true THEN 1 ELSE 0 END) as active,
                        AVG(CASE WHEN is_active = true THEN ai_score ELSE NULL END) as avg_score
                    FROM support_programs
                """)
                stats = cur.fetchone()
                
                # 최근 프로그램
                cur.execute("""
                    SELECT id, title, ai_score, created_at
                    FROM support_programs
                    WHERE is_active = true
                    ORDER BY created_at DESC
                    LIMIT 5
                """)
                recent = cur.fetchall()
                
                return {
                    'total_programs': stats[0] or 0,
                    'active_programs': stats[1] or 0,
                    'average_ai_score': round(stats[2] or 0, 1),
                    'recent_programs': [
                        {
                            'id': r[0],
                            'title': r[1],
                            'ai_score': r[2],
                            'created_at': r[3].isoformat() if r[3] else None
                        }
                        for r in recent
                    ]
                }
    
    def close(self):
        """연결 풀 종료"""
        if hasattr(self, 'connection_pool'):
            self.connection_pool.closeall()
            logger.info("PostgreSQL 연결 풀 종료")


# 싱글톤 인스턴스
db = DatabaseManager()