"""The Intel GPU driver check (av1sfm.devices.check_xpu_driver)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from av1sfm.devices import check_xpu_driver


@pytest.fixture
def driver(monkeypatch):
    """Set the driver version that torch.xpu reports."""

    def set_version(version: str) -> None:
        props = SimpleNamespace(driver_version=version)
        monkeypatch.setattr(torch.xpu, "get_device_properties", lambda *_: props, raising=False)

    return set_version


@pytest.mark.parametrize("version", ["1.17.39395+14", "1.18.40000", "unknown"])
def test_new_or_unknown_driver_passes(driver, version):
    driver(version)
    check_xpu_driver(torch.device("xpu"))


def test_old_driver_raises(driver):
    driver("1.14.37020")
    with pytest.raises(RuntimeError, match="too old"):
        check_xpu_driver(torch.device("xpu"))


def test_other_devices_are_not_checked(driver):
    driver("1.14.37020")
    check_xpu_driver(torch.device("cpu"))
