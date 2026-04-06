from abc import ABC, abstractmethod


class IndividualInitializer(ABC):
    @abstractmethod
    def generate_chrm():
        pass
