#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 응답 표준화를 위한 유틸리티 모듈
"""

from flask import jsonify
from typing import Any, Dict, Optional, Union


def success_response(
    data: Optional[Union[Dict, list]] = None,
    message: Optional[str] = None,
    status_code: int = 200
):
    """
    성공 응답 생성
    
    Args:
        data: 응답 데이터
        message: 성공 메시지
        status_code: HTTP 상태 코드
    
    Returns:
        Flask Response 객체
    """
    response = {
        "success": True,
        "data": data or {},
        "message": message
    }
    
    # None 값 제거
    response = {k: v for k, v in response.items() if v is not None}
    
    return jsonify(response), status_code


def error_response(
    error: str,
    message: Optional[str] = None,
    status_code: int = 400,
    details: Optional[Dict] = None
):
    """
    에러 응답 생성
    
    Args:
        error: 에러 타입 또는 코드
        message: 에러 메시지
        status_code: HTTP 상태 코드
        details: 추가 에러 정보
    
    Returns:
        Flask Response 객체
    """
    response = {
        "success": False,
        "error": error,
        "message": message or error,
        "details": details
    }
    
    # None 값 제거
    response = {k: v for k, v in response.items() if v is not None}
    
    return jsonify(response), status_code


def paginated_response(
    data: list,
    page: int,
    per_page: int,
    total: int,
    message: Optional[str] = None
):
    """
    페이지네이션 응답 생성
    
    Args:
        data: 페이지 데이터
        page: 현재 페이지
        per_page: 페이지당 항목 수
        total: 전체 항목 수
        message: 추가 메시지
    
    Returns:
        Flask Response 객체
    """
    total_pages = (total + per_page - 1) // per_page if per_page > 0 else 0
    
    response_data = {
        "items": data,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "has_prev": page > 1,
            "has_next": page < total_pages
        }
    }
    
    return success_response(data=response_data, message=message)


def validation_error_response(errors: Dict[str, Any]):
    """
    검증 에러 응답 생성
    
    Args:
        errors: 필드별 에러 메시지
    
    Returns:
        Flask Response 객체
    """
    return error_response(
        error="VALIDATION_ERROR",
        message="입력값 검증 실패",
        status_code=422,
        details={"validation_errors": errors}
    )


def not_found_response(resource: str = "리소스"):
    """
    404 Not Found 응답 생성
    
    Args:
        resource: 찾을 수 없는 리소스 이름
    
    Returns:
        Flask Response 객체
    """
    return error_response(
        error="NOT_FOUND",
        message=f"{resource}를 찾을 수 없습니다",
        status_code=404
    )


def unauthorized_response(message: str = "인증이 필요합니다"):
    """
    401 Unauthorized 응답 생성
    
    Args:
        message: 에러 메시지
    
    Returns:
        Flask Response 객체
    """
    return error_response(
        error="UNAUTHORIZED",
        message=message,
        status_code=401
    )


def forbidden_response(message: str = "접근 권한이 없습니다"):
    """
    403 Forbidden 응답 생성
    
    Args:
        message: 에러 메시지
    
    Returns:
        Flask Response 객체
    """
    return error_response(
        error="FORBIDDEN",
        message=message,
        status_code=403
    )


def internal_server_error_response(
    message: str = "서버 내부 오류가 발생했습니다",
    details: Optional[Dict] = None
):
    """
    500 Internal Server Error 응답 생성
    
    Args:
        message: 에러 메시지
        details: 추가 에러 정보 (디버그용)
    
    Returns:
        Flask Response 객체
    """
    return error_response(
        error="INTERNAL_SERVER_ERROR",
        message=message,
        status_code=500,
        details=details
    )