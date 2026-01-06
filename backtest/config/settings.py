"""
Configuration management for the backtesting system
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os
from dotenv import load_dotenv


DEFAULT_CONFIG = {
    # Data
    "assets": ["BTCUSDT", "ETHUSDT"],
    "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
    "history_days": 730,  # 2 years

    # Wave Detection
    "zigzag_atr_multiplier": 1.5,
    "zigzag_atr_period": 20,
    "min_wave_size_pct": 0.5,

    # Entry
    "fib_tolerance_pct": 2.5,
    "entry_fib_levels": [0.5, 0.618, 0.786],
    "initial_position_pct": 50,

    # Risk
    "base_risk_pct": 1.0,
    "initial_balance": 10000.0,
    "max_portfolio_heat_pct": 6.0,
    "module_a_risk_mult": 1.0,
    "module_b_risk_mult": 0.75,
    "module_c_risk_mult": 0.5,

    # Exits
    "tp1_pct": 40,
    "tp2_pct": 60,
    "stop_behavior": "breakeven_after_tp1",  # original, breakeven, trailing

    # Fees
    "maker_fee_pct": 0.02,
    "taker_fee_pct": 0.04,

    # Validation
    "train_test_split": 0.7,

    # Modules
    "module_a_enabled": True,
    "module_b_enabled": True,
    "module_c_enabled": True,
    "trade_diagonals": False,

    # Module C sub-patterns
    "module_c_zigzag_enabled": True,
    "module_c_flat_enabled": True,
    "module_c_triangle_enabled": True,

    # Module C parameters
    "zigzag_b_min_retrace": 0.382,  # Wave B minimum retracement of A
    "zigzag_b_max_retrace": 0.786,  # Wave B maximum retracement of A
    "flat_b_min_retrace": 0.90,     # Flat: B retraces at least 90% of A
    "flat_b_max_retrace": 1.38,     # Expanded flat: B can exceed A by up to 138%
    "triangle_min_waves": 5,        # A-B-C-D-E

    # Optimization
    "optimization_trials": 200,
    "optimization_metric": "sharpe_ratio",
}


@dataclass
class Config:
    """Configuration class for backtest parameters"""

    # Data
    assets: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"])
    history_days: int = 730

    # Wave Detection
    zigzag_atr_multiplier: float = 1.5
    zigzag_atr_period: int = 20
    min_wave_size_pct: float = 0.5

    # Entry
    fib_tolerance_pct: float = 2.5
    entry_fib_levels: List[float] = field(default_factory=lambda: [0.5, 0.618, 0.786])
    initial_position_pct: float = 50.0

    # Risk
    base_risk_pct: float = 1.0
    initial_balance: float = 10000.0
    max_portfolio_heat_pct: float = 6.0
    module_a_risk_mult: float = 1.0
    module_b_risk_mult: float = 0.75
    module_c_risk_mult: float = 0.5

    # Exits
    tp1_pct: float = 40.0
    tp2_pct: float = 60.0
    stop_behavior: str = "breakeven_after_tp1"

    # Fees
    maker_fee_pct: float = 0.02
    taker_fee_pct: float = 0.04

    # Validation
    train_test_split: float = 0.7

    # Modules
    module_a_enabled: bool = True
    module_b_enabled: bool = True
    module_c_enabled: bool = True
    trade_diagonals: bool = False

    # Module C sub-patterns
    module_c_zigzag_enabled: bool = True
    module_c_flat_enabled: bool = True
    module_c_triangle_enabled: bool = True

    # Module C parameters
    zigzag_b_min_retrace: float = 0.382
    zigzag_b_max_retrace: float = 0.786
    flat_b_min_retrace: float = 0.90
    flat_b_max_retrace: float = 1.38
    triangle_min_waves: int = 5

    # Optimization
    optimization_trials: int = 200
    optimization_metric: str = "sharpe_ratio"

    # API (loaded from env)
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

    def __post_init__(self):
        """Load API credentials from environment"""
        load_dotenv()
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create Config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "assets": self.assets,
            "timeframes": self.timeframes,
            "history_days": self.history_days,
            "zigzag_atr_multiplier": self.zigzag_atr_multiplier,
            "zigzag_atr_period": self.zigzag_atr_period,
            "min_wave_size_pct": self.min_wave_size_pct,
            "fib_tolerance_pct": self.fib_tolerance_pct,
            "entry_fib_levels": self.entry_fib_levels,
            "initial_position_pct": self.initial_position_pct,
            "base_risk_pct": self.base_risk_pct,
            "initial_balance": self.initial_balance,
            "max_portfolio_heat_pct": self.max_portfolio_heat_pct,
            "module_a_risk_mult": self.module_a_risk_mult,
            "module_b_risk_mult": self.module_b_risk_mult,
            "module_c_risk_mult": self.module_c_risk_mult,
            "tp1_pct": self.tp1_pct,
            "tp2_pct": self.tp2_pct,
            "stop_behavior": self.stop_behavior,
            "maker_fee_pct": self.maker_fee_pct,
            "taker_fee_pct": self.taker_fee_pct,
            "train_test_split": self.train_test_split,
            "module_a_enabled": self.module_a_enabled,
            "module_b_enabled": self.module_b_enabled,
            "module_c_enabled": self.module_c_enabled,
            "trade_diagonals": self.trade_diagonals,
            "module_c_zigzag_enabled": self.module_c_zigzag_enabled,
            "module_c_flat_enabled": self.module_c_flat_enabled,
            "module_c_triangle_enabled": self.module_c_triangle_enabled,
            "zigzag_b_min_retrace": self.zigzag_b_min_retrace,
            "zigzag_b_max_retrace": self.zigzag_b_max_retrace,
            "flat_b_min_retrace": self.flat_b_min_retrace,
            "flat_b_max_retrace": self.flat_b_max_retrace,
            "triangle_min_waves": self.triangle_min_waves,
            "optimization_trials": self.optimization_trials,
            "optimization_metric": self.optimization_metric,
        }
