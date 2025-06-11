#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 DB의 사이트들 분석 및 지원사업 사이트 활성화
"""

import requests
import json

def analyze_db_sites():
    """기존 DB 사이트들 분석"""
    try:
        # API 호출
        response = requests.get('http://localhost:5001/api/debug_sites')
        data = response.json()

        print('=== 기존 DB 분석 ===')
        print(f'전체 사이트: {data["total_sites"]}개')
        print(f'활성화된 사이트: {data["enabled_sites_count"]}개')
        print()

        print('=== 현재 활성화된 사이트들 ===')
        for site in data['enabled_sites_detail']:
            print(f'✅ {site["name"]}: {site["url"]}')

        print()
        print('=== 비활성화 사이트 중 지원사업 관련 사이트들 ===')
        support_keywords = [
            '창업', '지원', '벤처', 'startup', '기업', '산업', '기술', '혁신', '투자',
            'k-startup', 'tips', 'kotra', 'sba', '중소기업', '스타트업', '테크노파크',
            '진흥원', 'kibo', 'kised', 'kosme', '신용보증', '정책금융'
        ]
        
        found_sites = []
        for site in data['sites_detail']:
            if not site['enabled']:
                name = site['name'].lower()
                url = site['url'].lower()
                if any(keyword in name for keyword in support_keywords) or \
                   any(keyword in url for keyword in support_keywords):
                    found_sites.append(site)

        print(f'발견된 지원사업 관련 사이트: {len(found_sites)}개')
        for i, site in enumerate(found_sites[:20]):  # 처음 20개만 출력
            print(f'{i+1:2d}. ❌ {site["name"]}: {site["url"]}')
            
        return found_sites

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return []

if __name__ == "__main__":
    analyze_db_sites() 