"""Client for fetching data from the acquisition service."""

import requests
from typing import Dict, Any, Optional


class DataAcquisitionClient:
    """Client for interacting with the data acquisition service."""

    def __init__(self, base_url: str, jwt_token: Optional[str] = None):
        self.base_url = base_url
        self.jwt_token = jwt_token
        self.endpoints = {
            "acquisition": f"{base_url}/api/acquisition",
            "ingestion": f"{base_url}/api/ingestion",
            "processing": f"{base_url}/api/processing",
        }

    def set_jwt_token(self, token: str):
        """Set the JWT token for authentication."""
        self.jwt_token = token

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers including authentication token."""
        headers = {}
        if self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"
        return headers

    def fetch_realtime_data(self) -> Dict[str, Any]:
        """Fetch real-time PDU data from the acquisition service."""
        try:
            response = requests.get(
                f"{self.endpoints['acquisition']}/realtime", headers=self._get_headers(), timeout=15
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error fetching real-time data: {e}")
        return {}

    def fetch_historical_data(
        self, start_date: str, end_date: str, time_unit: str = "day"
    ) -> Dict[str, Any]:
        """Fetch historical aggregated PDU data."""
        try:
            params = {"startDate": start_date, "endDate": end_date}

            if time_unit == "hour":
                params["today"] = "true"
            elif time_unit == "day":
                params["week"] = "true"
            elif time_unit == "week":
                params["month"] = "true"

            response = requests.get(
                f"{self.endpoints['acquisition']}/aggregate",
                params=params,
                headers=self._get_headers(),
                timeout=15,
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                print("Authentication required for aggregate data endpoint")
            else:
                print(
                    f"Aggregate data request failed with status "
                    f"{response.status_code}: {response.text}"
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error fetching aggregate data: {e}")
        return {}

    def fetch_pdu_logs(self, start_time: int, end_time: int) -> Dict[str, Any]:
        """Fetch PDU logs within a time range."""
        try:
            params = {"startTime": start_time, "endTime": end_time}
            response = requests.get(
                f"{self.endpoints['acquisition']}/realtime/logs",
                params=params,
                headers=self._get_headers(),
                timeout=15,
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error fetching PDU logs: {e}")
        return {}
