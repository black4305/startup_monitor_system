// AI 지원사업 모니터링 시스템 - 대시보드 JavaScript

// 전역 변수
let currentProgramCount = 0;
let lastUpdateTime = "";

// 초기화
document.addEventListener('DOMContentLoaded', function() {
    // HTML data 속성에서 초기값 읽기
    const container = document.querySelector('.container[data-recent-count]');
    if (container) {
        currentProgramCount = parseInt(container.dataset.recentCount) || 0;
        lastUpdateTime = container.dataset.lastUpdate || "";
    }
    
    // 30초마다 데이터 상태 확인
    setInterval(checkDataStatus, 30000);
    
    // 페이지 로드 시 즉시 확인
    setTimeout(checkDataStatus, 2000);
});

// 실시간 데이터 상태 확인
function checkDataStatus() {
    fetch('/api/data_status')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const newCount = data.total_programs;
            
            // 새로운 데이터가 추가되었는지 확인
            if (newCount > currentProgramCount) {
                console.log(`🆕 새로운 지원사업 발견: ${newCount - currentProgramCount}개`);
                showRefreshIndicator();
                updateStats(data);
                currentProgramCount = newCount;
            }
            
            // 상태 업데이트
            updateDataStatus('실시간 모니터링', 'success');
        }
    })
    .catch(error => {
        console.error('데이터 상태 확인 실패:', error);
        updateDataStatus('연결 오류', 'danger');
    });
}

// 데이터 상태 배지 업데이트
function updateDataStatus(text, status) {
    const statusElement = document.getElementById('dataStatus');
    if (statusElement) {
        statusElement.textContent = text;
        statusElement.className = `badge bg-${status} ms-2`;
    }
}

// 새로고침 알림 표시
function showRefreshIndicator() {
    const indicator = document.getElementById('refreshIndicator');
    if (indicator) {
        indicator.style.display = 'block';
        indicator.classList.add('pulse');
    }
}

// 통계 업데이트
function updateStats(data) {
    updateElement('totalPrograms', data.total_programs);
    updateElement('activePrograms', data.total_programs);
    updateElement('recentCount', Math.min(10, data.total_programs));
    
    // 카드에 새로운 데이터 애니메이션 추가
    const cards = document.querySelectorAll('.stat-card');
    cards.forEach(card => {
        card.classList.add('pulse');
        setTimeout(() => card.classList.remove('pulse'), 2000);
    });
}

// 요소 업데이트 헬퍼
function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

// 수동 새로고침
function manualRefresh() {
    const btn = document.getElementById('refreshBtn');
    if (!btn) return;
    
    const originalHtml = btn.innerHTML;
    
    btn.innerHTML = '<i class="fas fa-sync-alt fa-spin"></i> 새로고침 중...';
    btn.disabled = true;
    
    fetch('/api/refresh_data', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            setTimeout(() => location.reload(), 500);
        } else {
            showAlert('새로고침 실패: ' + data.error, 'danger');
        }
    })
    .catch(error => {
        console.error('새로고침 실패:', error);
        showAlert('새로고침 중 오류가 발생했습니다.', 'danger');
    })
    .finally(() => {
        btn.innerHTML = originalHtml;
        btn.disabled = false;
    });
}

// 프로그램 삭제
function deleteProgram(programId) {
    if (confirm('이 프로그램을 AI 학습 데이터로 사용하시겠습니까?')) {
        fetch(`/api/delete/${programId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const row = document.getElementById(`row-${programId}`);
                if (row) {
                    row.style.display = 'none';
                }
                showAlert(data.message || 'AI가 학습했습니다.', 'success');
                // 프로그램 수 업데이트
                currentProgramCount--;
                updateElement('recentCount', Math.max(0, currentProgramCount));
            } else {
                showAlert('AI 학습 실패: ' + (data.error || '알 수 없는 오류'), 'danger');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showAlert('AI 학습 중 오류가 발생했습니다.', 'danger');
        });
    }
}

// 알림 메시지 표시
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show alert-custom`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // 5초 후 자동 제거
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
}

// 스크롤 탑 버튼 (옵션)
window.addEventListener('scroll', function() {
    const scrollBtn = document.querySelector('.scroll-top-btn');
    if (scrollBtn) {
        if (window.pageYOffset > 300) {
            scrollBtn.classList.add('show');
        } else {
            scrollBtn.classList.remove('show');
        }
    }
});

function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}