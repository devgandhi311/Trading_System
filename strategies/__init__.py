# strategies: trading strategy definitions and signal generation
"""Strategy plan generation and selection."""

from strategies.base_strategy import BaseStrategy
from strategies.broken_wing_butterfly import BrokenWingButterflyResearchStrategy
from strategies.calendar_diagonal import CalendarSpreadStrategy, DiagonalSpreadStrategy
from strategies.credit_spread import CreditSpreadStrategy
from strategies.debit_spread import DebitSpreadStrategy
from strategies.expected_move_credit_spread import ExpectedMoveCreditSpreadResearchStrategy, ExpectedMoveCreditSpreadStrategy
from strategies.expected_move_iron_condor import ExpectedMoveIronCondorResearchStrategy
from strategies.iv_mean_reversion import IVMeanReversionPremiumResearchStrategy
from strategies.iron_condor import IronCondorStrategy
from strategies.long_gamma import LongGammaStrategy
from strategies.short_strangle import ShortStrangleStrategy
from strategies.strategy_research_base import BaseOptionsResearchStrategy, StrategyResearchResult
from strategies.strategy_selector import StrategySelector
from strategies.strategy_selector_v2 import StrategySelectionV2Result, StrategySelectorV2
from strategies.volatility_expansion import VolatilityExpansionResearchStrategy

__all__ = [
    "BaseStrategy",
    "BrokenWingButterflyResearchStrategy",
    "CalendarSpreadStrategy",
    "CreditSpreadStrategy",
    "DebitSpreadStrategy",
    "DiagonalSpreadStrategy",
    "BaseOptionsResearchStrategy",
    "ExpectedMoveCreditSpreadResearchStrategy",
    "ExpectedMoveCreditSpreadStrategy",
    "ExpectedMoveIronCondorResearchStrategy",
    "IronCondorStrategy",
    "IVMeanReversionPremiumResearchStrategy",
    "LongGammaStrategy",
    "ShortStrangleStrategy",
    "StrategySelector",
    "StrategySelectionV2Result",
    "StrategySelectorV2",
    "StrategyResearchResult",
    "VolatilityExpansionResearchStrategy",
]
