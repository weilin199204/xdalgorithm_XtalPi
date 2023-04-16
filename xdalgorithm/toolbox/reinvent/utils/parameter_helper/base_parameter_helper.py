import os
import json
import logging
from abc import ABC, abstractmethod

class BaseParameterHelper(ABC):
    """
    A basic class of parameter helper.

    """
    JSON_TEMPLATE = {}
    
    @abstractmethod
    def generate_template(self):
        return NotImplementedError("generate_template method is not implemented")