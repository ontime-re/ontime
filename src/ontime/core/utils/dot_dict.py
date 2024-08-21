class DotDict(dict):
    """
    A dictionary that allows access to its keys as attributes.
    """

    def __getattr__(self, attr):
        """
        Get an attribute

        :param attr: the attribute to get
        :return: the attribute
        """
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        """
        Set an attribute

        :param attr: the attribute to set
        :param value: the value to set as the attribute
        :return: None
        """
        self[attr] = value

    def __delattr__(self, attr):
        """
        Delete an attribute

        :param attr: the attribute to delete
        :return: None
        """
        try:
            del self[attr]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")
