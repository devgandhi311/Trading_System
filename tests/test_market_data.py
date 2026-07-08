from market.instrument_master import InstrumentMaster
from market.option_chain import OptionChain


def test_instrument_master_download_save_load_and_lookup(tmp_path) -> None:
    master = InstrumentMaster(storage_dir=tmp_path)

    records = master.download()
    loaded = master.load()

    assert records
    assert loaded
    assert (tmp_path / "instrument_master.csv").exists()
    assert master.lookup_by_symbol("NIFTY") is not None
    assert master.lookup_by_token("NIFTY") is not None
    assert master.lookup_options(underlying="NIFTY", option_type="CE")


def test_option_chain_helpers_and_snapshot(tmp_path) -> None:
    master = InstrumentMaster(storage_dir=tmp_path)
    master.download()
    chain = OptionChain(master)

    nearest = chain.get_nearest_expiry("NIFTY")
    monthly = chain.get_monthly_expiry("NIFTY")
    custom = chain.get_custom_expiry("NIFTY", nearest)
    snapshot = chain.build_option_chain_snapshot(underlying="NIFTY", spot_price=25010.0, expiry=custom)
    output = chain.save_option_chain_snapshot(snapshot, tmp_path / "option_chain.json")

    assert nearest <= monthly
    assert custom == nearest
    assert chain.get_atm_strike(25010.0, 50) == 25000.0
    assert chain.select_otm_strike(spot_price=25000.0, option_type="CE", strike_step=50) == 25050.0
    assert chain.select_itm_strike(spot_price=25000.0, option_type="PE", strike_step=50) == 25050.0
    assert snapshot.contracts
    assert output.exists()
