"""
웹 크롤링 모듈 - 지원사업 공고 수집 (강화 버전)
SSL, 인코딩, User-Agent 문제 해결
"""

import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
from typing import List, Dict, Optional
from datetime import datetime
import traceback
import chardet
import ssl
import time

from .config import Config

logger = logging.getLogger(__name__)

class WebCrawler:
    """웹 크롤링 엔진 - 강화 버전"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
        # 기본 세션 설정
        self.session = requests.Session()
        
        # User-Agent 로테이션 (봇 차단 우회)
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0'
        ]
        self.current_ua_index = 0
        
        # 향상된 헤더 설정
        self.update_session_headers()
        
        # SSL 컨텍스트 설정
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        # 인코딩 시도 순서
        self.encodings = ['utf-8', 'euc-kr', 'cp949', 'iso-8859-1']
        
        logger.info("🌐 웹 크롤러 초기화 완료")
    
    def update_session_headers(self):
        """세션 헤더 업데이트 (User-Agent 로테이션)"""
        self.session.headers.update({
            'User-Agent': self.user_agents[self.current_ua_index],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1'
        })
        self.current_ua_index = (self.current_ua_index + 1) % len(self.user_agents)
    
    def get_page_with_fallback(self, url: str, site_name: str) -> Optional[BeautifulSoup]:
        """강화된 페이지 가져오기 (SSL, 인코딩, User-Agent 문제 해결)"""
        logger.info(f"🌐 페이지 접속 시도: {site_name} - {url}")
        
        # 여러 방법으로 시도
        methods = [
            self._get_with_ssl_verification,  # 기본 SSL 검증
            self._get_without_ssl_verification,  # SSL 검증 비활성화
            self._get_with_different_ua,  # 다른 User-Agent
        ]
        
        for i, method in enumerate(methods):
            try:
                logger.debug(f"   시도 {i+1}: {method.__name__}")
                response = method(url)
                
                if response and response.status_code == 200:
                    # 인코딩 감지 및 처리
                    soup = self._parse_with_encoding_detection(response)
                    if soup:
                        logger.info(f"✅ {site_name} 접속 성공 (방법 {i+1})")
                        return soup
                        
            except Exception as e:
                logger.debug(f"   방법 {i+1} 실패: {e}")
                continue
        
        logger.warning(f"❌ {site_name} 모든 접속 방법 실패")
        return None
    
    def _get_with_ssl_verification(self, url: str):
        """SSL 검증 활성화 상태로 요청"""
        self.session.verify = True
        return self.session.get(url, timeout=Config.TIMEOUT)
    
    def _get_without_ssl_verification(self, url: str):
        """SSL 검증 비활성화 상태로 요청"""
        self.session.verify = False
        # SSL 경고 메시지 숨기기
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        return self.session.get(url, timeout=Config.TIMEOUT)
    
    def _get_with_different_ua(self, url: str):
        """다른 User-Agent로 요청"""
        self.update_session_headers()  # User-Agent 변경
        self.session.verify = False  # SSL도 비활성화
        return self.session.get(url, timeout=Config.TIMEOUT)
    
    def _parse_with_encoding_detection(self, response) -> Optional[BeautifulSoup]:
        """인코딩 감지 및 파싱"""
        # 1차: Response의 encoding 사용
        if response.encoding and response.encoding.lower() != 'iso-8859-1':
            try:
                soup = BeautifulSoup(response.content, 'html.parser')
                if self._is_valid_korean_content(soup):
                    logger.debug(f"   인코딩 성공: {response.encoding}")
                    return soup
            except Exception as e:
                logger.debug(f"   Response encoding 실패: {e}")
        
        # 2차: chardet로 인코딩 감지
        try:
            detected = chardet.detect(response.content)
            if detected['encoding'] and detected['confidence'] > 0.7:
                decoded_content = response.content.decode(detected['encoding'])
                soup = BeautifulSoup(decoded_content, 'html.parser')
                if self._is_valid_korean_content(soup):
                    logger.debug(f"   인코딩 감지 성공: {detected['encoding']} (신뢰도: {detected['confidence']:.2f})")
                    return soup
        except Exception as e:
            logger.debug(f"   Chardet 감지 실패: {e}")
        
        # 3차: 순차적 인코딩 시도
        for encoding in self.encodings:
            try:
                decoded_content = response.content.decode(encoding, errors='ignore')
                soup = BeautifulSoup(decoded_content, 'html.parser')
                if self._is_valid_korean_content(soup):
                    logger.debug(f"   인코딩 시도 성공: {encoding}")
                    return soup
            except Exception as e:
                logger.debug(f"   {encoding} 시도 실패: {e}")
                continue
        
        # 4차: 마지막 시도 (오류 무시)
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
            logger.debug(f"   기본 파싱 사용 (인코딩 무시)")
            return soup
        except Exception as e:
            logger.error(f"   모든 파싱 방법 실패: {e}")
            return None
    
    def _is_valid_korean_content(self, soup: BeautifulSoup) -> bool:
        """한국어 콘텐츠 유효성 검사"""
        try:
            text = soup.get_text()[:500]  # 처음 500자만 확인
            # 한글 문자가 있고, 깨진 문자가 적은지 확인
            korean_chars = len([c for c in text if '가' <= c <= '힣'])
            broken_chars = text.count('')
            
            return korean_chars > 10 and broken_chars < 5
        except:
            return False
    
    def load_sites_from_supabase(self, priority: str = None, region: str = None) -> List[Dict]:
        """Supabase에서 크롤링 대상 사이트 로드"""
        try:
            return self.db_manager.get_crawling_sites(
                enabled_only=True, 
                priority=priority, 
                region=region
            )
        except Exception as e:
            logger.error(f"❌ 사이트 로드 실패: {e}")
            return []
    
    async def crawl_websites(self, max_sites: int = None, priority: str = None, 
                           region: str = None, concurrent_batch: int = 3, progress_callback=None) -> List[Dict]:
        """배치 병렬 웹사이트 크롤링 (3개씩 동시 처리)"""
        sites = self.load_sites_from_supabase(priority, region)
        
        if max_sites:
            sites = sites[:max_sites]
            
        logger.info(f"🔍 {len(sites)}개 사이트에서 배치 병렬 크롤링 시작 (3개씩 동시)")
        
        results = []
        completed_count = 0
        
        # 3개씩 배치로 나누기
        batch_size = 3
        for batch_start in range(0, len(sites), batch_size):
            batch_sites = sites[batch_start:batch_start + batch_size]
            
            logger.info(f"📦 배치 {batch_start//batch_size + 1}: {len(batch_sites)}개 사이트 동시 처리")
            
            # 배치 내 사이트들을 병렬로 처리
            batch_results = await self._process_batch_parallel(batch_sites, progress_callback, completed_count, len(sites))
            results.extend(batch_results)
            
            completed_count += len(batch_sites)
            
            # 배치 간 지연 (서버 부하 방지)
            if batch_start + batch_size < len(sites):
                logger.info("⏱️ 배치 간 2초 대기...")
                await asyncio.sleep(2)
        
        # 최종 완료 상태 업데이트
        if progress_callback:
            progress_callback(
                completed=len(sites), 
                total=len(sites), 
                current_site="",
                message="크롤링 완료"
            )
        
        logger.info(f"✅ 배치 병렬 크롤링 완료: {len(results)}개 프로그램 수집")
        return results
    
    async def _process_batch_parallel(self, batch_sites: List[Dict], progress_callback, completed_so_far: int, total_sites: int) -> List[Dict]:
        """배치 내 사이트들을 병렬 처리"""
        import concurrent.futures
        import time
        
        batch_results = []
        
        # ThreadPoolExecutor로 동시 처리 (3개씩) - 에러 처리 강화
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # 각 사이트별 Future 생성
                future_to_site = {}
                
                for i, site in enumerate(batch_sites):
                    site_name = site.get('name', '')
                    logger.info(f"🎯 사이트 {completed_so_far + i + 1}/{total_sites}: {site_name} (배치 처리)")
                    
                    # 진행 상태 업데이트
                    if progress_callback:
                        progress_callback(
                            completed=completed_so_far + i, 
                            total=total_sites, 
                            current_site=site_name,
                            message=f"배치 처리 중: {site_name}"
                        )
                    
                    # 동시 실행 시작
                    future = executor.submit(self.crawl_single_site, site)
                    future_to_site[future] = site
                
                # 결과 수집 (완료되는 순서대로) - 타임아웃 처리 개선
                try:
                    for future in concurrent.futures.as_completed(future_to_site, timeout=150):
                        site = future_to_site[future]
                        site_name = site.get('name', '')
                        
                        try:
                            site_programs = future.result(timeout=30)  # 개별 타임아웃
                            if site_programs:
                                batch_results.extend(site_programs)
                                logger.info(f"✅ {site_name}: {len(site_programs)}개 수집 (배치 완료)")
                            else:
                                logger.warning(f"⚠️ {site_name}: 프로그램 없음 (배치 완료)")
                                
                        except concurrent.futures.TimeoutError:
                            logger.warning(f"⚠️ {site_name} 타임아웃 - 건너뜀")
                            future.cancel()
                        except Exception as e:
                            logger.error(f"❌ {site_name} 배치 크롤링 실패: {e}")
                            continue
                            
                except concurrent.futures.TimeoutError:
                    logger.error("⚠️ 배치 전체 타임아웃 - 완료된 작업만 수집")
                    # 완료된 Future들만 결과 수집
                    for future, site in future_to_site.items():
                        if future.done() and not future.cancelled():
                            try:
                                site_programs = future.result(timeout=1)
                                if site_programs:
                                    batch_results.extend(site_programs)
                                    logger.info(f"✅ {site.get('name')}: {len(site_programs)}개 수집 (타임아웃 구조)")
                            except:
                                pass
                                
        except Exception as e:
            logger.error(f"❌ 배치 처리 중 심각한 오류: {e}")
            # 배치 실패 시 순차 처리로 폴백
            logger.info("🔄 순차 처리로 폴백...")
            for site in batch_sites:
                site_name = site.get('name', '')
                try:
                    site_programs = self.crawl_single_site(site)
                    if site_programs:
                        batch_results.extend(site_programs)
                        logger.info(f"✅ {site_name}: {len(site_programs)}개 수집 (순차 폴백)")
                except Exception as site_e:
                    logger.error(f"❌ {site_name} 순차 크롤링도 실패: {site_e}")
                    continue
        
        logger.info(f"📦 배치 완료: {len(batch_results)}개 프로그램 수집")
        return batch_results
    
    async def crawl_websites_streaming(self, max_sites: int = None, priority: str = None, 
                                     region: str = None, concurrent_batch: int = 3,
                                     program_callback=None, site_progress_callback=None):
        """실시간 스트리밍 크롤링 (프로그램 발견 즉시 콜백 호출)"""
        try:
            logger.info("🔄 실시간 스트리밍 크롤링 시작")
            
            # 사이트 목록 로드
            sites = self.load_sites_from_supabase(priority=priority, region=region)
            if max_sites:
                sites = sites[:max_sites]
            
            logger.info(f"📋 스트리밍 크롤링 대상: {len(sites)}개 사이트")
            
            completed_sites = 0
            total_programs_found = 0
            
            # 배치별 스트리밍 크롤링
            for i in range(0, len(sites), concurrent_batch):
                batch_sites = sites[i:i + concurrent_batch]
                
                logger.info(f"📦 스트리밍 배치 {i//concurrent_batch + 1}: {len(batch_sites)}개 사이트 처리")
                
                # 배치 내 사이트들을 병렬로 처리하면서 실시간 콜백
                batch_results = await self._process_batch_streaming(
                    batch_sites, program_callback, site_progress_callback, 
                    completed_sites, len(sites)
                )
                
                # 통계 업데이트
                for site_programs_count in batch_results:
                    total_programs_found += site_programs_count
                
                completed_sites += len(batch_results)
                
                # 배치 간 지연
                if i + concurrent_batch < len(sites):
                    logger.info("⏱️ 스트리밍 배치 간 1초 대기...")
                    await asyncio.sleep(1)
            
            logger.info(f"✅ 스트리밍 크롤링 완료: {completed_sites}개 사이트에서 총 {total_programs_found}개 프로그램 처리")
            
        except Exception as e:
            logger.error(f"❌ 스트리밍 크롤링 실패: {e}")
            raise
    
    async def _process_batch_streaming(self, batch_sites: List[Dict], program_callback, 
                                     site_progress_callback, completed_so_far: int, total_sites: int) -> List[int]:
        """배치 내 사이트들을 병렬 처리하면서 프로그램 발견 즉시 콜백 호출"""
        import concurrent.futures
        
        batch_results = []
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # 각 사이트별 Future 생성
                future_to_site = {}
                
                for i, site in enumerate(batch_sites):
                    site_name = site.get('name', '')
                    logger.info(f"🎯 스트리밍 사이트 {completed_so_far + i + 1}/{total_sites}: {site_name}")
                    
                    # 스트리밍용 사이트 크롤링 실행
                    future = executor.submit(self._crawl_single_site_streaming, site, program_callback)
                    future_to_site[future] = site
                
                # 결과 수집 (완료되는 순서대로)
                try:
                    for future in concurrent.futures.as_completed(future_to_site, timeout=120):
                        site = future_to_site[future]
                        site_name = site.get('name', '')
                        
                        try:
                            programs_count = future.result(timeout=30)
                            batch_results.append(programs_count)
                            
                            # 사이트 완료 진행 상황 콜백
                            if site_progress_callback:
                                await site_progress_callback(
                                    site_info=site,
                                    programs_found=programs_count,
                                    completed_sites=completed_so_far + len(batch_results),
                                    total_sites=total_sites
                                )
                            
                            logger.info(f"✅ {site_name}: {programs_count}개 처리 완료 (스트리밍)")
                            
                        except concurrent.futures.TimeoutError:
                            logger.warning(f"⚠️ {site_name} 스트리밍 타임아웃 - 건너뜀")
                            batch_results.append(0)
                            future.cancel()
                        except Exception as e:
                            logger.error(f"❌ {site_name} 스트리밍 크롤링 실패: {e}")
                            batch_results.append(0)
                            continue
                            
                except concurrent.futures.TimeoutError:
                    logger.error("⚠️ 스트리밍 배치 전체 타임아웃")
                    # 미완료 사이트들은 0으로 처리
                    while len(batch_results) < len(batch_sites):
                        batch_results.append(0)
                        
        except Exception as e:
            logger.error(f"❌ 스트리밍 배치 처리 중 오류: {e}")
            # 실패 시 0으로 채우기
            while len(batch_results) < len(batch_sites):
                batch_results.append(0)
        
        logger.info(f"📦 스트리밍 배치 완료: {sum(batch_results)}개 프로그램 처리")
        return batch_results
    
    def _crawl_single_site_streaming(self, site_info: Dict, program_callback) -> int:
        """단일 사이트 스트리밍 크롤링 (프로그램 발견 즉시 콜백)"""
        site_name = site_info.get('name', '')
        programs_count = 0
        
        try:
            logger.info(f"🔍 스트리밍 크롤링: {site_name}")
            
            # 기존 크롤링 로직 사용
            programs = self.crawl_single_site(site_info)
            
            if programs and program_callback:
                # 발견된 각 프로그램을 즉시 콜백으로 전달
                for program_data in programs:
                    try:
                        # 비동기 콜백을 동기 환경에서 실행
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # 콜백 실행 (AI 분석 + DB 저장)
                        success = loop.run_until_complete(
                            program_callback(program_data, site_info)
                        )
                        
                        if success:
                            programs_count += 1
                            logger.debug(f"📤 프로그램 스트리밍 성공: '{program_data.get('title', '')[:30]}...'")
                        else:
                            logger.debug(f"📤 프로그램 스트리밍 실패: '{program_data.get('title', '')[:30]}...'")
                            
                        loop.close()
                        
                    except Exception as callback_error:
                        logger.error(f"❌ 프로그램 콜백 실패: {callback_error}")
                        continue
            
            logger.info(f"✅ {site_name}: {programs_count}개 스트리밍 완료")
            return programs_count
            
        except Exception as e:
            logger.error(f"❌ {site_name} 스트리밍 크롤링 실패: {e}")
            return 0
    
    async def crawl_single_site_async(self, session: aiohttp.ClientSession, 
                                    semaphore: asyncio.Semaphore, site_info: Dict) -> List[Dict]:
        """단일 사이트 비동기 크롤링 (강화 버전)"""
        async with semaphore:
            try:
                site_id = site_info.get('id', '')
                site_name = site_info.get('name', '')
                base_url = site_info.get('url', '')
                
                logger.info(f"🔍 강화된 크롤링: {site_name}")
                
                # SSL 검증 비활성화된 커넥터 사용
                connector = aiohttp.TCPConnector(ssl=False)
                timeout = aiohttp.ClientTimeout(total=Config.TIMEOUT)
                
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    # 헤더 설정
                    headers = {
                        'User-Agent': self.user_agents[0],
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                    }
                    
                    async with session.get(base_url, headers=headers) as response:
                        if response.status != 200:
                            logger.warning(f"⚠️ {site_name} 접속 실패: {response.status}")
                            return []
                        
                        html_content = await response.read()
                        
                        # 인코딩 처리
                        soup = self._parse_response_content(html_content)
                        if not soup:
                            logger.warning(f"⚠️ {site_name} 파싱 실패")
                            return []
                        
                        programs = self.extract_announcements_from_site(soup, site_info)
                        
                        # 크롤링 통계 업데이트
                        self.db_manager.update_crawling_stats(site_id, True, {
                            'programs_found': len(programs),
                            'crawled_at': datetime.now().isoformat()
                        })
                        
                        logger.info(f"✅ {site_name}: {len(programs)}개 프로그램 수집")
                        return programs
                        
            except Exception as e:
                logger.error(f"❌ {site_name} 크롤링 실패: {e}")
                self.db_manager.update_crawling_stats(site_id, False, {
                    'error': str(e),
                    'crawled_at': datetime.now().isoformat()
                })
                return []
    
    def _parse_response_content(self, content: bytes) -> Optional[BeautifulSoup]:
        """Response 내용 파싱 (인코딩 감지)"""
        # chardet로 인코딩 감지
        try:
            detected = chardet.detect(content)
            if detected['encoding'] and detected['confidence'] > 0.7:
                decoded_content = content.decode(detected['encoding'])
                return BeautifulSoup(decoded_content, 'html.parser')
        except:
            pass
        
        # 순차적 인코딩 시도
        for encoding in self.encodings:
            try:
                decoded_content = content.decode(encoding, errors='ignore')
                return BeautifulSoup(decoded_content, 'html.parser')
            except:
                continue
        
        return None
    
    def crawl_single_site(self, site_info: Dict) -> List[Dict]:
        """단일 사이트 동기 크롤링 (강화 버전)"""
        site_name = site_info.get('name', '')
        base_url = site_info.get('url', '')
        
        logger.info(f"🔍 강화된 크롤링: {site_name}")
        
        try:
            # 강화된 페이지 가져오기
            soup = self.get_page_with_fallback(base_url, site_name)
            
            if not soup:
                logger.warning(f"⚠️ {site_name} 페이지 로드 실패")
                return []
            
            programs = self.extract_announcements_from_site(soup, site_info)
            logger.info(f"✅ {site_name}: {len(programs)}개 수집")
            return programs
            
        except Exception as e:
            logger.error(f"❌ {site_name} 크롤링 실패: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def extract_announcements_from_site(self, soup: BeautifulSoup, site_info: Dict) -> List[Dict]:
        """사이트에서 공고 추출 - 다중 페이지 지원 버전"""
        site_name = site_info.get('name', '')
        base_url = site_info.get('url', '')
        
        programs = []
        logger.info(f"🔍 {site_name} 에서 공고 추출 시작")
        
        # 현재 페이지에서 프로그램 추출
        current_page_programs = self._extract_programs_from_page(soup, site_info, base_url)
        programs.extend(current_page_programs)
        
        # 여러 페이지 처리 (최대 3페이지까지)
        max_pages = 3
        page_count = 1
        
        # 페이지네이션 링크 찾기
        pagination_links = self._find_pagination_links(soup, base_url)
        
        for page_url in pagination_links[:max_pages-1]:  # 현재 페이지 제외하고 2페이지 더
            try:
                logger.info(f"📄 {site_name} {page_count+1}페이지 크롤링: {page_url}")
                
                page_soup = self.get_page_with_fallback(page_url, site_name)
                if page_soup:
                    page_programs = self._extract_programs_from_page(page_soup, site_info, page_url)
                    programs.extend(page_programs)
                    page_count += 1
                    
                    # 요청 간격 조절
                    time.sleep(1)
                else:
                    logger.warning(f"⚠️ {site_name} {page_count+1}페이지 로드 실패")
                    
            except Exception as e:
                logger.warning(f"⚠️ {site_name} {page_count+1}페이지 처리 실패: {e}")
                continue
        
        # 중복 제거
        unique_programs = []
        seen_urls = set()
        
        for program in programs:
            if program['url'] not in seen_urls:
                unique_programs.append(program)
                seen_urls.add(program['url'])
        
        logger.info(f"✅ {site_name}: 총 {len(unique_programs)}개 프로그램 수집 ({page_count}페이지)")
        return unique_programs[:Config.MAX_PROGRAMS_PER_SITE]
    
    def _find_pagination_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """페이지네이션 링크 찾기"""
        pagination_links = []
        
        # 다양한 페이지네이션 패턴 시도
        pagination_selectors = [
            # 일반적인 페이지네이션
            '.pagination a', '.paging a', '.page-nav a',
            # 번호 기반
            'a[href*="page="]', 'a[href*="pageNo="]', 'a[href*="p="]',
            # 다음 페이지
            'a[href*="next"]', '.next a', '.btn-next',
            # 한국어 다음
            'a:contains("다음")', 'a:contains(">")', 'a:contains("▶")'
        ]
        
        for selector in pagination_selectors:
            try:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href and href not in pagination_links:
                        full_url = urljoin(base_url, href)
                        pagination_links.append(full_url)
                        
                        # 최대 5개 링크까지만
                        if len(pagination_links) >= 5:
                            break
                            
                if pagination_links:
                    break  # 하나의 패턴에서 찾으면 중단
                    
            except Exception as e:
                continue
        
        return pagination_links
    
    def _extract_programs_from_page(self, soup: BeautifulSoup, site_info: Dict, page_url: str) -> List[Dict]:
        """단일 페이지에서 프로그램 추출"""
        site_name = site_info.get('name', '')
        programs = []
        
        # 1단계: 모든 링크에서 지원사업 관련 링크 찾기
        all_links = soup.find_all('a', href=True)
        
        # 지원사업 관련 키워드 (URL 및 텍스트)
        support_keywords = [
            # 한국어 키워드
            '공고', '지원', '사업', '모집', '신청', '선정', '창업', '벤처', 
            '스타트업', '기업', '투자', '자금', '보조금', '융자', 'r&d', '연구', '개발',
            # 영어 키워드  
            'notice', 'support', 'business', 'program', 'startup', 'venture',
            'funding', 'grant', 'investment', 'announce', 'call', 'apply'
        ]
        
        found_links = []
        for link in all_links:
            href = link.get('href', '').lower()
            text = link.get_text(strip=True).lower()
            
            # URL이나 텍스트에 지원사업 키워드가 포함된 경우
            if any(keyword in href for keyword in support_keywords) or \
               any(keyword in text for keyword in support_keywords):
                found_links.append(link)
        
        # 중복 제거
        unique_links = {}
        for link in found_links:
            href = link.get('href', '')
            if href and href not in unique_links:
                unique_links[href] = link
        
        for href, link in list(unique_links.items()):
            try:
                # 상대경로를 절대경로로 변환
                full_url = urljoin(page_url, href)
                title = link.get_text(strip=True)
                
                if self.is_support_program_title(title):
                    # 간단한 내용 추출
                    content = self.extract_brief_content(full_url, title)
                    
                    program = {
                        'title': title,
                        'content': content,
                        'url': full_url,
                        'site_name': site_name,
                        'extracted_at': datetime.now().isoformat()
                    }
                    programs.append(program)
                    
            except Exception as e:
                logger.warning(f"링크 처리 실패 {href}: {e}")
                continue
        
        return programs
    
    def extract_brief_content(self, url: str, title: str) -> str:
        """URL에서 상세한 공고 내용 추출 (AI 분석용 - 강화 버전)"""
        try:
            # 상세 페이지 접속
            page_soup = self.get_page_with_fallback(url, title[:20])
            if not page_soup:
                logger.warning(f"⚠️ 상세 페이지 접속 실패: {url}")
                return "상세 내용을 가져올 수 없습니다."
            
            # 불필요한 요소 제거
            for unwanted in page_soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'comment']):
                unwanted.decompose()
            
            extracted_content = []
            
            # 1단계: 제목 다시 추출 (페이지 내 제목이 더 정확할 수 있음)
            title_selectors = [
                'h1', 'h2', '.title', '.subject', '.board-title', '.notice-title',
                '.view-title', '.content-title', '.detail-title', 'article h1'
            ]
            for selector in title_selectors:
                title_elem = page_soup.select_one(selector)
                if title_elem and len(title_elem.get_text().strip()) > 5:
                    page_title = title_elem.get_text().strip()
                    if page_title != title and len(page_title) > 10:  # 기존 제목과 다르고 충분히 길면
                        extracted_content.append(f"📋 상세제목: {page_title}")
                    break
            
            # 2단계: 본문 내용 추출 (더 포괄적인 선택자)
            content_selectors = [
                # 일반적인 콘텐츠 선택자
                '.content', '.board-content', '.notice-content', '.view-content',
                '#content', '#board_content', '#notice_content', '#view_content',
                '.post-content', '.article-content', '.text-content', '.detail-content',
                
                # 테이블 기반 콘텐츠
                '.board-view', '.view-body', '.content-body', '.detail-body',
                'table.view', 'table.board', '.board_view', '#board_view',
                
                # 기타 가능한 선택자
                'article', '.article', 'main', '.main', '#main',
                '.container .content', '.wrapper .content', '.board-wrap'
            ]
            
            main_content = ""
            for selector in content_selectors:
                content_elem = page_soup.select_one(selector)
                if content_elem:
                    # 테이블이면 구조적으로 파싱
                    if content_elem.name == 'table' or content_elem.find('table'):
                        main_content = self._parse_table_content(content_elem)
                    else:
                        main_content = content_elem.get_text(separator='\n', strip=True)
                    
                    if len(main_content) > 100:  # 충분한 내용이 있으면
                        break
            
            # 3단계: 본문이 없으면 전체 body에서 핵심 정보 추출
            if not main_content or len(main_content) < 200:
                body_text = page_soup.get_text(separator='\n', strip=True)
                main_content = self._extract_relevant_content(body_text)
            
            if main_content:
                # 핵심 정보 추출을 위한 구조화된 파싱
                lines = main_content.split('\n')
                important_info = {}
                
                # 중요한 정보 패턴
                info_patterns = {
                    '지원금액': r'(지원금액|지원규모|사업비|지원한도)[\s:：]*(.*?)(?:\n|$)',
                    '지원대상': r'(지원대상|신청자격|대상기업|참가자격)[\s:：]*(.*?)(?:\n|$)',
                    '신청기간': r'(신청기간|접수기간|모집기간|공고기간)[\s:：]*(.*?)(?:\n|$)',
                    '지원내용': r'(지원내용|사업내용|주요내용|지원사항)[\s:：]*(.*?)(?:\n|$)',
                    '선정기준': r'(선정기준|평가기준|심사기준|선발기준)[\s:：]*(.*?)(?:\n|$)'
                }
                
                import re
                for key, pattern in info_patterns.items():
                    match = re.search(pattern, main_content, re.MULTILINE | re.DOTALL)
                    if match:
                        value = match.group(2).strip()
                        if value and len(value) > 5:
                            important_info[key] = value[:300]  # 최대 300자
                
                # 구조화된 내용 생성
                if important_info:
                    for key, value in important_info.items():
                        extracted_content.append(f"\n🔸 {key}: {value}")
                
                # 금액 정보 특별 추출
                money_matches = re.findall(r'(\d+(?:,\d+)*(?:\.\d+)?\s*(?:억원|천만원|백만원|만원|원))', main_content)
                if money_matches:
                    extracted_content.append(f"\n💰 금액정보: {', '.join(set(money_matches[:5]))}")
                
                # 날짜 정보 특별 추출
                date_matches = re.findall(r'(\d{4}[\s년.-]\d{1,2}[\s월.-]\d{1,2}[\s일]?)', main_content)
                if date_matches:
                    extracted_content.append(f"\n📅 일정정보: {', '.join(set(date_matches[:5]))}")
                
                # 전체 내용 요약 (최대 1500자)
                if len(extracted_content) < 3:  # 구조화된 정보가 부족하면
                    summary = self._summarize_content(main_content, 1500)
                    extracted_content.append(f"\n📄 내용요약:\n{summary}")
            
            # 최종 컨텐츠 조합
            if extracted_content:
                return '\n'.join(extracted_content)
            else:
                return f"제목: {title}\n(상세 내용 추출 실패)"
                
        except Exception as e:
            logger.error(f"❌ 상세 내용 추출 실패 {url}: {e}")
            return f"제목: {title}\n(상세 내용 로드 오류)"
    
    def is_support_program_title(self, title: str) -> bool:
        """제목이 지원사업인지 판단"""
        if not title or len(title.strip()) < 5:
            return False
        
        # 지원사업 관련 키워드
        support_keywords = [
            '지원', '사업', '공고', '모집', '선정', '신청',
            '창업', '벤처', '스타트업', '기업', '투자',
            'R&D', '연구', '개발', '기술', '혁신'
        ]
        
        # 제외할 키워드
        exclude_keywords = [
            '채용', '구인', '구직', '입찰', '공사',
            '행사', '세미나', '강의', '교육'
        ]
        
        title_lower = title.lower()
        
        # 제외 키워드 체크
        for keyword in exclude_keywords:
            if keyword in title_lower:
                return False
        
        # 지원사업 키워드 체크
        for keyword in support_keywords:
            if keyword in title_lower:
                return True
        
        return False
    
    def _parse_table_content(self, table_elem) -> str:
        """테이블 형식의 콘텐츠를 구조적으로 파싱"""
        content_lines = []
        
        # 테이블의 모든 행 처리
        rows = table_elem.find_all('tr')
        for row in rows:
            # th와 td 요소 추출
            headers = row.find_all('th')
            cells = row.find_all('td')
            
            # 헤더가 있으면 헤더: 값 형식으로
            if headers and cells:
                for i, header in enumerate(headers):
                    if i < len(cells):
                        header_text = header.get_text(strip=True)
                        cell_text = cells[i].get_text(strip=True)
                        if header_text and cell_text:
                            content_lines.append(f"{header_text}: {cell_text}")
            # 헤더가 없으면 셀 내용만
            elif cells:
                row_text = ' | '.join([cell.get_text(strip=True) for cell in cells])
                if row_text.strip():
                    content_lines.append(row_text)
        
        return '\n'.join(content_lines)
    
    def _extract_relevant_content(self, full_text: str) -> str:
        """전체 텍스트에서 지원사업 관련 내용만 추출"""
        lines = full_text.split('\n')
        relevant_lines = []
        
        # 관련 키워드
        relevant_keywords = [
            '지원', '사업', '공고', '모집', '신청', '대상', '자격', '기간',
            '금액', '규모', '내용', '조건', '선정', '평가', '제출', '서류',
            '창업', '스타트업', '벤처', '기업', '투자', '융자', '보조금',
            '억원', '천만원', '백만원', '만원', '기한', '마감', '접수'
        ]
        
        # 각 줄을 검사하여 관련 내용만 추출
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in relevant_keywords):
                relevant_lines.append(line.strip())
                
                # 충분한 내용을 수집했으면 중단
                if len('\n'.join(relevant_lines)) > 1000:
                    break
        
        return '\n'.join(relevant_lines)
    
    def _summarize_content(self, content: str, max_length: int = 1500) -> str:
        """긴 콘텐츠를 요약"""
        if len(content) <= max_length:
            return content
        
        # 문장 단위로 분리
        sentences = content.replace('。', '.').split('.')
        
        # 중요한 문장 우선 선택
        important_sentences = []
        other_sentences = []
        
        important_keywords = [
            '지원금액', '지원규모', '지원대상', '신청기간', '마감일',
            '선정기준', '평가기준', '지원내용', '사업내용', '지원조건',
            '억원', '천만원', '백만원', '만원'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 중요 키워드를 포함한 문장 우선
            if any(keyword in sentence for keyword in important_keywords):
                important_sentences.append(sentence)
            else:
                other_sentences.append(sentence)
        
        # 중요 문장부터 채우고, 남은 공간에 다른 문장 추가
        result = []
        current_length = 0
        
        # 중요 문장 추가
        for sentence in important_sentences:
            if current_length + len(sentence) < max_length:
                result.append(sentence)
                current_length += len(sentence) + 1
        
        # 나머지 문장 추가
        for sentence in other_sentences:
            if current_length + len(sentence) < max_length:
                result.append(sentence)
                current_length += len(sentence) + 1
            else:
                break
        
        return '. '.join(result) + '...' 