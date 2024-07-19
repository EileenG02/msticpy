# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Intsights Provider.

Input can be a single IoC observable or a pandas DataFrame containing
multiple observables. Processing may require a an API key and
processing performance may be limited to a specific number of
requests per minute for the account type that you have.
"""
import datetime as dt
from typing import Any, Dict, Tuple

import attr

from ..._version import VERSION
from ...common.utility import export
from ..http_provider import APILookupParams
from .result_severity import ResultSeverity
from .ti_http_provider import HttpTIProvider

__version__ = VERSION
__author__ = "Florian Bracq"


_DEF_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}


# pylint: disable=too-few-public-methods
@attr.s
class _IntSightsParams(APILookupParams):
    # override APILookupParams to set common defaults
    def __attrs_post_init__(self):
        self.auth_str = ["{ApiID}", "{AuthKey}"]
        self.auth_type = "HTTPBasic"


@export
class IntSights(HttpTIProvider):
    """IntSights Lookup."""

    _BASE_URL = "https://api.ti.insight.rapid7.com"

    _QUERIES = {
        "ipv4": _IntSightsParams(
            path="/public/v3/iocs/ioc-by-value",
            params={"iocValue": "{observable}"},
            headers=_DEF_HEADERS,
        ),
        "ipv6": _IntSightsParams(
            path="/public/v3/iocs/ioc-by-value",
            params={"iocValue": "{observable}"},
            headers=_DEF_HEADERS,
        ),
        "dns": _IntSightsParams(
            path="/public/v3/iocs/ioc-by-value",
            params={"iocValue": "{observable}"},
            headers=_DEF_HEADERS,
        ),
        "url": _IntSightsParams(
            path="/public/v3/iocs/ioc-by-value",
            params={"iocValue": "{observable}"},
            headers=_DEF_HEADERS,
        ),
        "md5_hash": _IntSightsParams(
            path="/public/v3/iocs/ioc-by-value",
            params={"iocValue": "{observable}"},
            headers=_DEF_HEADERS,
        ),
        "sha1_hash": _IntSightsParams(
            path="/public/v3/iocs/ioc-by-value",
            params={"iocValue": "{observable}"},
            headers=_DEF_HEADERS,
        ),
        "sha256_hash": _IntSightsParams(
            path="/public/v3/iocs/ioc-by-value",
            params={"iocValue": "{observable}"},
            headers=_DEF_HEADERS,
        ),
        "email": _IntSightsParams(
            path="/public/v3/iocs/ioc-by-value",
            params={"iocValue": "{observable}"},
            headers=_DEF_HEADERS,
        ),
    }

    _REQUIRED_PARAMS = ["ApiID", "AuthKey"]

    def parse_results(self, response: Dict) -> Tuple[bool, ResultSeverity, Any]:
        """
        Return the details of the response.

        Parameters
        ----------
        response : Dict
            The returned data response

        Returns
        -------
        Tuple[bool, ResultSeverity, Any]
            bool = positive or negative hit
            ResultSeverity = enumeration of severity
            Object with match details

        """
        if self._failed_response(response) or not isinstance(
            response["RawResult"], dict
        ):
            return False, ResultSeverity.information, "Not found."

        if response["RawResult"].get("whitelisted", False):
            return False, ResultSeverity.information, "Whitelisted."

        sev = response["RawResult"].get("severity", "Low")
        result_dict = {
            "threat_actors": response["RawResult"].get("relatedThreatActors", ""),
            "geolocation": response["RawResult"].get("geolocation", None),
            "response_code": response["Status"],
            "tags": response["RawResult"].get("tags", [])
            + response["RawResult"].get("SystemTags", []),
            "malware": response["RawResult"].get("relatedMalware", []),
            "campaigns": response["RawResult"].get("relatedCampaigns", []),
            "score": response["RawResult"].get("score", 0),
            "first_seen": dt.datetime.strptime(
                response["RawResult"].get("firstSeen", None), "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "last_seen": dt.datetime.strptime(
                response["RawResult"].get("lastSeen", None), "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "last_update": dt.datetime.strptime(
                response["RawResult"].get("lastUpdateDate", None),
                "%Y-%m-%dT%H:%M:%S.%fZ",
            ),
        }

        severity = (
            ResultSeverity.information
            if sev == "Low"
            else (
                ResultSeverity.warning
                if sev == "Medium"
                else ResultSeverity.high if sev == "High" else ResultSeverity.unknown
            )
        )

        return True, severity, result_dict
