"""Excel I/O contract tests (3.2 multi-sheet, 3.9 named sheet)."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import HAS_DATA

requires_data = pytest.mark.skipif(not HAS_DATA, reason='experiment archive not present')


def test_xlswrite_preserves_multiple_sheets(tmp_path):
    from model_compat import xlswrite_compat
    from openpyxl import load_workbook

    path = tmp_path / 'multi.xlsx'
    xlswrite_compat(str(path), np.array([[1.0], [2.0]]), 'a')
    xlswrite_compat(str(path), np.array([[3.0], [4.0]]), 'b')
    xlswrite_compat(str(path), np.array([[5.0]]), 'c')

    wb = load_workbook(path)
    assert {'a', 'b', 'c'}.issubset(set(wb.sheetnames))


def test_load_numeric_excel_named_sheet_and_missing():
    from model_compat import load_numeric_excel
    from paths import box_surfareas_with_centers_path

    path = str(box_surfareas_with_centers_path())
    arr = load_numeric_excel(path, sheet_name='Sheet1')
    assert arr.ndim == 2 and arr.shape[0] > 0
    with pytest.raises(KeyError):
        load_numeric_excel(path, sheet_name='does-not-exist')
