import numpy as np
import pytest
import cv2


# 샘플 함수: 가짜 깊이맵 생성
def generate_depth_map(image):
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 가짜 깊이맵 적용 (컬러맵)
    depth_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)
    return depth_map


# 테스트 코드
def test_generate_depth_map():
    image = np.zeros((100, 100, 3), dtype=np.uint8)  # 검정색 빈 이미지
    depth_map = generate_depth_map(image)
    assert depth_map.shape == image.shape, "출력 크기가 입력 크기와 다릅니다."
    assert isinstance(depth_map, np.ndarray), "출력 데이터 타입이 ndarray가 아닙니다."


def test_depth_map_not_none():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    depth_map = generate_depth_map(image)


def test_depth_map_none():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    depth_map = generate_depth_map(image)

    # 일부러 실패: None이어야 통과되게 작성
    assert depth_map is None, "일부러 실패시키기: 출력이 None이 아님"


# pytest 실행
if __name__ == "__main__":
    pytest.main()
