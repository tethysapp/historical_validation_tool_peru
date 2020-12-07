from tethys_sdk.base import TethysAppBase, url_map_maker


class HistoricalValidationToolPeru(TethysAppBase):
    """
    Tethys app class for Historical Validation Tool Peru.
    """

    name = 'Historical Validation Tool Peru'
    index = 'historical_validation_tool_peru:home'
    icon = 'historical_validation_tool_peru/images/icon.gif'
    package = 'historical_validation_tool_peru'
    root_url = 'historical-validation-tool-peru'
    color = '#2980b9'
    description = 'This app evaluates the accuracy for the historical streamflow values obtained from Streamflow Prediction Tool in Peru.'
    tags = '"Hydrology"'
    enable_feedback = False
    feedback_emails = []

    def url_maps(self):
        """
        Add controllers
        """
        UrlMap = url_map_maker(self.root_url)

        url_maps = (
            UrlMap(
                name='home',
                url='historical-validation-tool-peru',
                controller='historical_validation_tool_peru.controllers.home'
            ),
        )

        return url_maps