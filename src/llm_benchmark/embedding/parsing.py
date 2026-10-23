"""파일 경계의 비정형 값을 검증해 내부 데이터 타입으로 좁힌다."""


def mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("Expected object with string keys")
    return {key: item for key, item in value.items()}


def string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected string")
    return value


def integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("Expected integer")
    return value


def sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Expected list")
    return list(value)
