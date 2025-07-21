// AI 지원사업 모니터링 시스템 - 프로그램 목록 JavaScript

let currentProgramId = null;
let autoDeleteCheckInterval = null;

// 프로그램 내용 토글
function toggleContent(programId) {
    const content = document.getElementById(`content-${programId}`);
    if (content) {
        if (content.classList.contains('show')) {
            content.classList.remove('show');
        } else {
            content.classList.add('show');
        }
    }
}

// 프로그램 삭제
function deleteProgram(programId) {
    currentProgramId = programId;
    
    // 라디오 버튼 초기화
    const radioButtons = document.querySelectorAll('input[name="deleteReason"]');
    radioButtons.forEach(radio => radio.checked = false);
    
    const modal = new bootstrap.Modal(document.getElementById('deleteReasonModal'));
    modal.show();
}

// 프로그램 관심 표시
function markInterested(programId) {
    if (confirm('이 프로그램에 관심이 있으신가요?')) {
        recordFeedback(programId, 'interested');
    }
}

// 피드백 기록
function recordFeedback(programId, action) {
    fetch(`/api/feedback/${programId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: action })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert(`AI가 학습했습니다: ${action === 'interested' ? '관심 있음' : '관심 없음'}`, 'success');
            
            if (action === 'interested') {
                const row = document.getElementById(`row-${programId}`);
                if (row) {
                    row.style.backgroundColor = '#e8f5e9';
                    setTimeout(() => {
                        row.style.backgroundColor = '';
                    }, 2000);
                }
            }
        } else {
            showAlert('피드백 저장 실패: ' + data.error, 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('피드백 저장 중 오류가 발생했습니다.', 'danger');
    });
}

// 삭제 확인 (confirmDeleteWithReason과 동일)
function confirmDelete() {
    confirmDeleteWithReason();
}

// 삭제 확인 (이유 포함)
function confirmDeleteWithReason() {
    const reason = document.querySelector('input[name="deleteReason"]:checked')?.value;
    
    if (!reason) {
        alert('삭제 이유를 선택해주세요.');
        return;
    }
    
    // 모달 닫기
    const modal = bootstrap.Modal.getInstance(document.getElementById('deleteReasonModal'));
    modal.hide();
    
    // 일괄 삭제인지 개별 삭제인지 확인
    if (currentProgramId === 'bulk') {
        processBulkDelete(reason);
    } else {
        processSingleDelete(currentProgramId, reason);
    }
}

// 개별 삭제 처리
function processSingleDelete(programId, reason) {
    fetch(`/api/delete/${programId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: reason })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById(`row-${programId}`).style.display = 'none';
            showAlert(`AI가 "${reason}" 이유로 학습했습니다. 더 정확한 추천을 위해 활용됩니다.`, 'success');
            
            // 자동 삭제 모니터링 시작
            startAutoDeleteMonitoring();
            
            // 페이지 새로고침은 자동 삭제 완료 후에
        } else {
            showAlert('AI 학습 실패: ' + (data.error || '알 수 없는 오류'), 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('AI 학습 중 오류가 발생했습니다.', 'danger');
    });
}

// 일괄 삭제 처리
function processBulkDelete(reason) {
    const selectedPrograms = Array.from(document.querySelectorAll('.program-checkbox:checked'))
        .map(checkbox => checkbox.value);
    
    fetch('/api/bulk-delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ program_ids: selectedPrograms, reason: reason })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            selectedPrograms.forEach(id => {
                const row = document.getElementById(`row-${id}`);
                if (row) {
                    row.style.display = 'none';
                }
            });
            
            showAlert(`AI가 ${selectedPrograms.length}개 항목을 "${reason}" 이유로 학습했습니다.`, 'success');
            
            // 선택 상태 초기화
            document.getElementById('selectAll').checked = false;
            updateSelectedCount();
            
            // 자동 삭제 모니터링 시작
            startAutoDeleteMonitoring();
        } else {
            showAlert('일괄 삭제 실패: ' + (data.error || '알 수 없는 오류'), 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('일괄 삭제 중 오류가 발생했습니다.', 'danger');
    });
    
    // 모달 제목 복원
    document.getElementById('deleteReasonModalLabel').innerHTML = 
        '<i class="fas fa-brain text-warning"></i> AI 학습 - 삭제 이유 선택';
}

// 전체 선택/해제
function toggleSelectAll() {
    const checkboxes = document.querySelectorAll('.program-checkbox');
    const selectAll = document.getElementById('selectAll').checked;
    
    checkboxes.forEach(checkbox => {
        checkbox.checked = selectAll;
    });
    
    updateSelectedCount();
}

// 선택된 항목 수 업데이트
function updateSelectedCount() {
    const checkboxes = document.querySelectorAll('.program-checkbox');
    const selectedCount = Array.from(checkboxes).filter(checkbox => checkbox.checked).length;
    
    document.getElementById('selectedCount').textContent = selectedCount;
    document.getElementById('bulkDeleteBtn').disabled = selectedCount === 0;
}

// 일괄 삭제 시작
function bulkDeletePrograms() {
    const selectedPrograms = Array.from(document.querySelectorAll('.program-checkbox:checked'))
        .map(checkbox => checkbox.value);
    
    if (selectedPrograms.length === 0) {
        alert('선택된 항목이 없습니다.');
        return;
    }
    
    // 일괄 삭제 모드 설정
    currentProgramId = 'bulk';
    
    // 모달 제목 변경
    document.getElementById('deleteReasonModalLabel').innerHTML = 
        `<i class="fas fa-brain text-warning"></i> AI 학습 - ${selectedPrograms.length}개 항목 삭제 이유 선택`;
    
    const modal = new bootstrap.Modal(document.getElementById('deleteReasonModal'));
    modal.show();
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

// 자동 삭제 모니터링 시작
function startAutoDeleteMonitoring() {
    // 모달 표시
    const modal = new bootstrap.Modal(document.getElementById('autoDeleteModal'));
    modal.show();
    
    // UI 비활성화
    disableUI();
    
    // 주기적으로 상태 확인 (0.5초마다)
    autoDeleteCheckInterval = setInterval(() => {
        checkAutoDeleteStatus();
    }, 500);
}

// 자동 삭제 상태 확인
function checkAutoDeleteStatus() {
    fetch('/api/auto_delete_status')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.data.status) {
                const status = data.data.status;
                
                // UI 업데이트
                updateAutoDeleteUI(status);
                
                // 완료 확인
                if (!status.is_running) {
                    // 인터벌 중지
                    clearInterval(autoDeleteCheckInterval);
                    
                    // UI 업데이트
                    completeAutoDelete(status);
                }
            }
        })
        .catch(error => {
            console.error('자동 삭제 상태 확인 오류:', error);
        });
}

// 자동 삭제 UI 업데이트
function updateAutoDeleteUI(status) {
    // 진행 상황 업데이트
    document.getElementById('triggerProgram').textContent = status.trigger_program || '-';
    document.getElementById('processedCount').textContent = status.processed || 0;
    document.getElementById('totalCount').textContent = status.total_programs || 0;
    document.getElementById('deletedCount').textContent = status.deleted || 0;
    document.getElementById('currentProgram').textContent = status.current_program || '-';
    
    // 프로그레스바 업데이트
    const progress = status.total_programs > 0 
        ? Math.round((status.processed / status.total_programs) * 100) 
        : 0;
    
    const progressBar = document.getElementById('progressBar');
    progressBar.style.width = progress + '%';
    document.getElementById('progressText').textContent = progress + '%';
}

// 자동 삭제 완료
function completeAutoDelete(status) {
    // 프로그레스바 100%로 설정
    const progressBar = document.getElementById('progressBar');
    progressBar.style.width = '100%';
    document.getElementById('progressText').textContent = '100%';
    
    // 애니메이션 제거
    progressBar.classList.remove('progress-bar-animated');
    progressBar.classList.add('bg-success');
    
    // 버튼 활성화
    const closeBtn = document.getElementById('closeModalBtn');
    closeBtn.textContent = '완료';
    closeBtn.classList.remove('btn-secondary');
    closeBtn.classList.add('btn-success');
    closeBtn.disabled = false;
    closeBtn.onclick = () => {
        // 모달 닫고 페이지 새로고침
        const modal = bootstrap.Modal.getInstance(document.getElementById('autoDeleteModal'));
        modal.hide();
        location.reload();
    };
    
    // 완료 메시지 표시
    if (status.deleted > 0) {
        showAlert(`AI가 총 ${status.deleted}개의 유사한 프로그램을 자동으로 삭제했습니다.`, 'success');
    } else {
        showAlert('유사한 프로그램을 찾지 못했습니다.', 'info');
    }
}

// UI 비활성화
function disableUI() {
    // 모든 버튼 비활성화
    document.querySelectorAll('button').forEach(btn => {
        if (!btn.id || btn.id !== 'closeModalBtn') {
            btn.disabled = true;
        }
    });
    
    // 체크박스 비활성화
    document.querySelectorAll('input[type="checkbox"]').forEach(cb => {
        cb.disabled = true;
    });
}

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    // 체크박스 이벤트 리스너 추가
    const checkboxes = document.querySelectorAll('.program-checkbox');
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', updateSelectedCount);
    });
    
    // 초기 선택 수 업데이트
    updateSelectedCount();
});