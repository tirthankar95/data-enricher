from abc import ABC, abstractmethod


class DataLoader(ABC):
    @abstractmethod
    def iterate(self):
        pass
    def impute(self, metadata: dict, new_data: str):
        pass


class DataFactoryRegistry:
    _registry = {}
    @classmethod
    def register(cls, key: str, target_cls):
        if key in cls._registry and cls._registry[key] != target_cls:
            raise ValueError(f'Class already registered {key=}, {target_cls=}')
        cls._registry[key] = target_cls
    @classmethod
    def get_data_loader(cls, key: str):
        if key not in cls._registry:
            raise ValueError(f'Class {key=} not yet registered.')
        return cls._registry[key]


def register(key: str):
    def decorator(cls):
        DataFactoryRegistry.register(key, cls)
        return cls
    return decorator
