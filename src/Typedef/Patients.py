# ./Typedef/Patients.py
import os
from typing import TypeAlias

PatientId: TypeAlias = str
MriImage: TypeAlias = os.PathLike
MaskImage: TypeAlias = os.PathLike

MriSegment: TypeAlias = tuple[MriImage, MaskImage]
Patients: TypeAlias = dict[PatientId, list[MriSegment]]

RejectedSegment: TypeAlias = tuple[os.PathLike, str]
RejectedSegments: TypeAlias = dict[PatientId, list[RejectedSegment]]