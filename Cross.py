
class Cross(object):
    """
    Class for cross containing date, time, red and blue values and positions.

    Provides eq based on date and time not on value, and to string for development.
    """

    def __init__(self, time, date, red_pos, red_value, blue_pos, blue_value):
        self.time = time
        self.date = date
        self.red_pos = red_pos
        self.red_value = red_value
        self.blue_pos = blue_pos
        self.blue_value = blue_value

    def __eq__(self, other):
        if isinstance(other, Cross):
            if other.time == self.time and self.date == other.date:
                return True
        return False

    def __str__(self):
        return 'Cross at time {} and date {}, with red value {} and blue value {}'.format(self.time, self.date,
                                                                                          self.red_value, self.blue_value)