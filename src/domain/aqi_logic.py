"""
Core domain logic for Air Quality Index (AQI).
"""

def get_aqi_category(aqi_value):
    """
    Returns the AQI category based on the value.
    """
    if aqi_value <= 50:
        return 'Good'
    if aqi_value <= 100:
        return 'Moderate'
    if aqi_value <= 150:
        return 'Unhealthy'
    return 'Hazardous'

def get_aqi_color(category):
    """
    Returns a color code for a specific AQI category.
    """
    colors = {
        'Good': '#00e400',
        'Moderate': '#ffff00',
        'Unhealthy': '#ff7e00',
        'Hazardous': '#7e0023'
    }
    return colors.get(category, '#cccccc')
