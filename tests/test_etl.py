import pytest

from retailpulse.data import etl


def test_etl_outputs_and_dedup(small_cfg, tables):
    tx = tables["transactions"]
    assert not tx.duplicated().any()
    assert {"revenue", "margin", "is_return", "is_guest", "category"} <= set(tx.columns)
    assert tx["is_return"].any() and (tx.loc[tx["is_return"], "quantity"] < 0).all()
    assert tx["is_guest"].equals(tx["customer_id"].isna())


def test_revenue_formula(tables):
    tx = tables["transactions"].head(500)
    expected = tx["quantity"] * tx["unit_price"] * (1 - tx["discount_pct"] / 100)
    assert (tx["revenue"] - expected).abs().max() < 1e-9


def test_validation_catches_broken_foreign_key(small_cfg):
    from retailpulse.config import resolve_path
    frames = etl.extract(resolve_path(small_cfg, "raw_dir"))
    frames["transactions"].loc[0, "product_id"] = "NOPE"
    rep = etl.validate(frames)
    assert not rep.ok
    assert any(c["check"] == "tx:fk_product" and not c["passed"] for c in rep.checks)


def test_validation_catches_missing_columns(small_cfg):
    from retailpulse.config import resolve_path
    frames = etl.extract(resolve_path(small_cfg, "raw_dir"))
    frames["products"] = frames["products"].drop(columns=["category"])
    assert not etl.validate(frames).ok


def test_missing_raw_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        etl.extract(tmp_path)
