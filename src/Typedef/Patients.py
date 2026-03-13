# ./Typedef/Patients.py
import numpy as np
from typing import TypeAlias

PatientId: TypeAlias = str
MriImage: TypeAlias = np.ndarray
MaskImage: TypeAlias = np.ndarray

MriSegment: TypeAlias = tuple[MriImage, MaskImage]
Patients: TypeAlias = dict[PatientId, list[MriSegment]]