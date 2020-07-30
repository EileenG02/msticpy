# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""LogAnayltics Uploader class."""
import json
import datetime
import sys
import hashlib
import hmac
import base64
import re
from pathlib import Path

import requests
from tqdm.notebook import tqdm
import pandas as pd

from ...common.exceptions import MsticpyConnectionError

from .uploader_base import UploaderBase
from ..._version import VERSION

# Credits
# https://docs.microsoft.com/en-us/azure/azure-monitor/platform/data-collector-api#python-3-sample
# https://github.com/Cyb3rWard0g/azure-loganalytics-api-clients/blob/master/python/ala-python-data-producer.py


__version__ = VERSION
__author__ = "Pete Bryan"


class LAUploader(UploaderBase):
    """Uploader class for LogAnalytics."""

    def __init__(self, workspace: str, workspace_secret: str, **kwargs):
        """Initialize a LogAnalytics Uploader instance."""
        super().__init__()
        self._kwargs = kwargs
        self.workspace = workspace
        self.workspace_secret = workspace_secret
        self._debug = kwargs.get("debug", False)
        self.ops_loc = kwargs.get("opsinsight_loc", ".ods.opinsights.azure.com")

    def _build_signature(
        self,
        date: str,
        content_length: int,
        method: str,
        content_type: str,
        resource: str,
    ) -> str:
        """
        Build authentication authentication string to pass to LA API.

        Parameters
        ----------
        date : str
            datetime of authencation session.
        content_length : int
            lenght of content to be passed to the api.
        method : str
            HTTP method being used.
        content_type : str
            Type of content being passed to the API.
        resource : str
            The API endpoint being targetted.

        Returns
        -------
        str
            The encoded authorization string.

        """
        x_headers = "x-ms-date:" + date
        string_to_hash = (
            method
            + "\n"
            + str(content_length)
            + "\n"
            + content_type
            + "\n"
            + x_headers
            + "\n"
            + resource
        )
        bytes_to_hash = bytes(string_to_hash, encoding="utf-8")
        decoded_key = base64.b64decode(self.workspace_secret)
        encoded_hash = base64.b64encode(
            hmac.new(decoded_key, bytes_to_hash, digestmod=hashlib.sha256).digest()
        ).decode()
        authorization = f"SharedKey {self.workspace}:{encoded_hash}"
        return authorization

    def _post_data(self, body: str, table_name: str):
        """
        Write data to Log Analytics Workspace.

        Parameters
        ----------
        body : str
            The JSON formatted data to write to Log Analytics.
        table_name : str
            The name of the custom table to write the data to.

        Raises
        ------
        MsticpyConnectionError
            Raised when response code indicates failure.

        """
        table_name = re.sub("[^A-Za-z0-9_]+", "", table_name)

        resource = "/api/logs"
        content_type = "application/json"
        rfc1123date = datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
        content_length = len(body)
        signature = self._build_signature(
            rfc1123date, content_length, "POST", content_type, resource
        )
        uri = (
            "https://"
            + self.workspace
            + self.ops_loc
            + resource
            + "?api-version=2016-04-01"
        )
        headers = {
            "content-type": content_type,
            "Authorization": signature,
            "Log-Type": table_name,
            "x-ms-date": rfc1123date,
        }
        try:
            response = requests.post(uri, data=body, headers=headers)
        except requests.ConnectionError:
            raise MsticpyConnectionError(
                "Unable to connect to workspace, ensure your Workspace ID is correct.",
                title="Unable to connect to Workspace",
            )
        if self._debug is True:
            print(f"Upload response code: {response.status_code}")
        if response.status_code < 200 or response.status_code > 299:
            raise MsticpyConnectionError(
                f"""LogAnalytics data upload failed with code {response.status_code}.
                Check Workspace ID and key""",
                title="Data Upload Failed",
            )

    def upload_df(self, data: pd.DataFrame, table_name: str, **kwargs):
        """
        Upload a pandas DataFrame to Log Analytics.

        Parameters
        ----------
        data : pd.DataFrame
            Pandas DataFrame to upload.
        table_name : str
            Custom table name to upload the data to.

        """
        events = []
        for row in data.iterrows():
            events.append(row[1].astype(str).to_dict())
            # Due to 30MB limit if data is larger than 25Mb upload that chunk then continue
            if sys.getsizeof(json.dumps(events)) > 26214400:
                if self._debug is True:
                    print("Data larger than 25MB spliting data requests.")
                body = json.dumps(events)
                self._post_data(body, table_name)
                events = []

        if events:
            body = json.dumps(events)
            self._post_data(body, table_name)

        if self._debug:
            print(f"Upload to {table_name} complete")

    def upload_file(
        self, file_path: str, table_name: str = None, delim: str = ",", **kwargs,
    ):
        """
        Upload a seperated value file to Log Analytics.

        Parameters
        ----------
        file_path : str
            Path to file to upload.
        table_name : str
            Table name to upload data to.
        delim : str, optional
            Value seperator used by the file, by default ","

        """
        path = Path(file_path)
        data = pd.read_csv(path, delimiter=delim)
        if not table_name:
            table_name = str(path).split("\\")[-1].split(".")[0]
        self.upload_df(data, table_name)

    def upload_folder(
        self, folder_path: str, table_name: str = None, delim: str = ",", **kwargs
    ):
        """
        Upload all files in a folder to Log Analytics.

        Parameters
        ----------
        folder_path : str
            Folder path to upload.
        table_name : str, optional
            Table name to upload all files to, by default None.
        delim : str, optional
            Sperator used in files in target folder, by default ",".

        """
        if delim != ",":
            ext = "*"
        else:
            ext = "*.csv"
        t_name = bool(table_name)
        input_files = Path(folder_path).glob(ext)
        input_files = [
            path for path in input_files  # pylint: disable=unnecessary-comprehension
        ]
        progress = tqdm(total=len(input_files), desc="Files", position=0)
        for path in input_files:
            data = pd.read_csv(path, delimiter=delim)
            if t_name is False:
                table_name = str(path).split("\\")[-1].split(".")[0]
            self.upload_df(data, table_name)
            progress.update(1)
        progress.close()
