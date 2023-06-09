

class ScoringFunctionNameEnum():
    __CUSTOM_PRODUCT = "custom_product"
    __CUSTOM_SUM = "custom_sum"
    __CUSTOM_FILTERING = 'custom_filtering'

    @property
    def CUSTOM_PRODUCT(self):
        return self.__CUSTOM_PRODUCT

    @CUSTOM_PRODUCT.setter
    def CUSTOM_PRODUCT(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionNameEnum field")

    @property
    def CUSTOM_SUM(self):
        return self.__CUSTOM_SUM

    @CUSTOM_SUM.setter
    def CUSTOM_SUM(self, value):
        raise ValueError("Do not assign value to a ScoringFunctionNameEnum field")

    # @property
    # def CUSTOM_FILTERING(self):
    #     return self.__CUSTOM_FILTERING
    #
    # @CUSTOM_FILTERING.setter
    # def CUSTOM_FILTERING(self, value):
    #     raise ValueError("Do not assign value to a ScoringFunctionNameEnum field")