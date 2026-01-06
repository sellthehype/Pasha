# Visual Backtest Verification Tool - Specification

## Overview

A standalone HTML visualization tool for verifying the Elliott Wave backtest engine correctness. Uses ETHUSDT 1D data to display candlestick charts with overlaid wave patterns, trade entries/exits, and real-time backtest playback via timeline scrubbing.

---

## 1. Technology Stack

- **Output**: Single self-contained HTML file (~2-3MB with embedded data)
- **Charting**: Plotly.js (interactive candlestick charts)
- **Styling**: Dark theme (trading terminal aesthetic)
- **Data**: All OHLC, trades, pivots, waves embedded as inline JSON
- **No dependencies**: Opens directly in any modern browser

---

## 2. Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Header: ETHUSDT 1D Backtest Verification                    [Controls] │
├─────────────────────────────────────────────────────────────┬───────────┤
│                                                             │           │
│                                                             │  Trade    │
│                     Main Candlestick Chart                  │  Info     │
│                        (70% width)                          │  Panel    │
│                                                             │  (30%)    │
│                                                             │           │
├─────────────────────────────────────────────────────────────┤           │
│                     Equity Curve                            │           │
│               (with drawdown shading)                       │           │
├─────────────────────────────────────────────────────────────┤           │
│  ◄──────────────── Timeline Scrubber ───────────────────►   │           │
├─────────────────────────────────────────────────────────────┴───────────┤
│                         Trade Log Table                                  │
│              (sortable, filterable, click to highlight)                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Main Chart Area (70% width)
- Candlestick chart with OHLC data
- Default zoom: Last 6 months (~180 candles)
- Pan/zoom enabled via mouse drag and scroll
- Crosshair with price/date readout

### 2.2 Sidebar Panel (30% width)
- **Current Position**: When trade is open
  - Entry price, current price, unrealized P&L
  - Time in trade, distance to SL/TP
- **Selected Trade Detail**: When trade clicked
  - Full accounting breakdown
  - Entry/exit prices, fees, net P&L
  - Position sizing calculation
  - Risk % used
- **Toggle Controls**: Layer visibility switches

### 2.3 Equity Curve Subplot
- Line chart showing equity over time
- Red shaded areas for drawdown periods
- Synced x-axis with main chart

### 2.4 Timeline Scrubber
- Horizontal slider spanning full data range
- Drag to any point, view updates to show state at that moment
- Current date/time displayed

### 2.5 Trade Log Table
- Columns: #, Date, Type, Module, Entry, Exit, P&L %, P&L $, Duration
- Sortable by any column
- Click row to highlight trade on chart and zoom to it

---

## 3. Visual Elements

### 3.1 Candlesticks
- Standard OHLC representation
- Green: close > open
- Red: close < open
- Wicks for high/low

### 3.2 Pivot Points
- **Pivot location marker**: Small dot at the pivot price
- **Confirmation arrow**: Arrow from confirmation bar pointing back to pivot
- Purpose: Verify look-ahead bias fix (entries happen AFTER confirmation)
- Color: Yellow/gold for pivots

### 3.3 Elliott Wave Patterns

#### Wave Labels
- Numeric notation: 1, 2, 3, 4, 5 at wave points
- Positioned at pivot highs/lows
- Font: Bold, readable size

#### Wave Lines
- Connect pivot points showing wave structure
- Dashed lines for in-progress waves
- Solid lines for completed waves

#### Pattern Distinction
- **Traded patterns**: Full opacity, thicker lines
- **Non-traded patterns**: 40% opacity, thinner lines

### 3.4 Fibonacci Levels (Toggleable Layers)

| Layer | Contents |
|-------|----------|
| Entry Zone | 38.2% - 78.6% retracement band (Module A) or 23.6% - 50% (Module B) |
| TP1 Level | 100% extension line |
| TP2 Level | 161.8% extension line |
| Stop Loss | Line at invalidation point |

- Each layer independently toggleable
- Active fibs: Full opacity
- Historical fibs: Faded (20% opacity)

### 3.5 Trade Markers

#### Entry Markers
- **Module A (Wave 3)**: Blue circle with "A" label
- **Module B (Wave 5)**: Orange circle with "B" label
- Arrow pointing to exact entry candle

#### Exit Markers
- **TP1 hit**: Green checkmark (partial, 40%)
- **TP2 hit**: Green double-checkmark (remaining 60%)
- **Stop Loss**: Red X marker
- **Breakeven stop**: Yellow dash

#### Trade Lifecycle Lines (toggleable)
- Default: Connected path from entry → TP1 → TP2/SL
- Toggle: Simplify to discrete markers only
- Line color matches trade outcome (green=profit, red=loss)

### 3.6 Failed Setups (Toggleable)
- Patterns detected but entry zone never reached
- Displayed with dotted lines, gray color
- Entry zone shown as faded band
- Label: "No entry" annotation

---

## 4. Interactive Features

### 4.1 Timeline Scrubbing
- Drag scrubber to any bar
- Chart updates to show:
  - All data up to that point (future hidden or faded)
  - Open positions with real-time MTM P&L
  - Currently active wave patterns
  - Visible Fib levels for active setups

### 4.2 Mark-to-Market Updates
- Open positions update unrealized P&L every bar
- Sidebar shows: current price, P&L %, distance to SL/TP

### 4.3 Click Interactions
- **Click trade marker**: Select trade, show full detail in sidebar
- **Click trade log row**: Highlight trade on chart, zoom to context
- **Hover candle**: Show OHLC tooltip
- **Hover trade**: Show quick summary

### 4.4 Keyboard Navigation
- **Left/Right arrows**: Step to previous/next trade
- Trade is highlighted and chart pans to center it

### 4.5 Toggle Controls (Sidebar)
| Toggle | Default | Description |
|--------|---------|-------------|
| Wave patterns | ON | Show all detected wave structures |
| Traded only | OFF | Hide non-traded patterns |
| Failed setups | OFF | Show patterns where entry wasn't reached |
| Entry zones | ON | Show Fib retracement entry bands |
| TP levels | ON | Show TP1/TP2 extension lines |
| Stop levels | ON | Show SL lines |
| Trade paths | ON | Show connected entry→exit lines |
| Pivot arrows | ON | Show confirmation→pivot arrows |

---

## 5. Audit Mode

### 5.1 Automatic Checks (Background)
Run on data load, surface issues if found:

1. **Data Integrity**
   - Verify OHLC consistency (high >= low, etc.)
   - Check for gaps in timestamp sequence
   - Validate price ranges are reasonable

2. **Entry Timing Verification**
   - Flag if any entry bar < confirmed_idx (look-ahead violation)
   - Highlight trades where timing seems suspicious

3. **Price Accuracy**
   - Verify entry price matches open of entry bar
   - Verify exit prices match actual OHLC at exit bar
   - Flag any discrepancies

4. **Wave Rule Validation**
   - Wave 2 doesn't retrace beyond Wave 1 start
   - Wave 4 doesn't overlap Wave 1 territory
   - Proper Fibonacci relationships

### 5.2 Visual Indicators
- **Subtle approach**: Small warning icons (⚠️) on flagged items
- Don't dominate view, but visible on inspection
- Hover for issue description
- Issues also listed in separate "Audit" tab in sidebar

---

## 6. Trade Information Panel (Sidebar)

### 6.1 Position Summary (when trade open)
```
LONG ETHUSDT
Entry: $2,450.00 @ Bar 523
Current: $2,512.00
Unrealized: +$62.00 (+2.53%)

Position: 0.408 ETH ($1,000.00)
Risk: 1.0% ($100.00)

Stop Loss: $2,350.00 (-4.08%)
TP1: $2,580.00 (+5.31%) - 40%
TP2: $2,680.00 (+9.39%) - 60%

Distance to SL: -$100
Distance to TP1: +$68
```

### 6.2 Completed Trade Detail (when trade selected)
```
TRADE #24 - Module A (Wave 3)
═══════════════════════════

Entry: 2024-06-15 @ $2,450.00
Exit:  2024-07-02 @ TP1 $2,580.00 (40%)
       2024-07-08 @ TP2 $2,680.00 (60%)

Duration: 23 days

Position Sizing:
  Balance at entry: $12,450.00
  Risk amount: $124.50 (1.0%)
  Stop distance: 4.08%
  Position size: 0.408 ETH ($1,000.00)

P&L Breakdown:
  TP1 (40%): +$53.06
  TP2 (60%): +$138.00
  Gross P&L: +$191.06

  Entry fee: -$0.80 (0.04%)
  TP1 fee:   -$0.41 (0.02%)
  TP2 fee:   -$0.54 (0.02%)
  Total fees: -$1.75

  Net P&L: +$189.31 (+18.93%)

Wave Context:
  Wave 1: $2,200 → $2,600
  Wave 2: $2,600 → $2,450 (38.5% retrace)
  Entry zone: $2,447 - $2,514
  Confirmed: Bar 520 (3 bars before entry)
```

---

## 7. Data Requirements

### 7.1 Input Data (embedded in HTML)
From Python backtest run, export:

```javascript
const DATA = {
  // OHLC candles
  candles: [
    {date: "2024-01-01", open: 2400, high: 2450, low: 2380, close: 2420},
    ...
  ],

  // All detected pivots with confirmation timing
  pivots: [
    {idx: 45, type: "high", price: 2600, confirmed_idx: 48},
    {idx: 52, type: "low", price: 2450, confirmed_idx: 55},
    ...
  ],

  // All detected wave patterns
  patterns: [
    {
      id: 1,
      type: "wave3",
      module: "A",
      pivots: [45, 52, 63],  // W1 end, W2 end, (W3 in progress)
      entry_zone: {low: 2447, high: 2514},
      tp1: 2580,
      tp2: 2680,
      stop: 2350,
      traded: true,
      trade_id: 24
    },
    ...
  ],

  // All trades with full detail
  trades: [
    {
      id: 24,
      module: "A",
      pattern_id: 1,
      direction: "long",
      entry_bar: 56,
      entry_price: 2450.00,
      quantity: 0.408,

      tp1_bar: 72,
      tp1_price: 2580.00,
      tp1_qty: 0.163,  // 40%

      tp2_bar: 78,
      tp2_price: 2680.00,
      tp2_qty: 0.245,  // 60%

      sl_bar: null,
      sl_price: null,

      fees: {entry: 0.80, tp1: 0.41, tp2: 0.54},
      pnl_gross: 191.06,
      pnl_net: 189.31,

      balance_before: 12450.00,
      balance_after: 12639.31,

      // For verification
      confirmed_bar: 53,  // When entry signal was valid
      pivot_bar: 52       // Where pivot actually occurred
    },
    ...
  ],

  // Equity curve
  equity: [
    {bar: 0, equity: 10000.00, drawdown: 0},
    {bar: 1, equity: 10000.00, drawdown: 0},
    ...
  ],

  // Audit flags
  audit: {
    data_issues: [],
    timing_issues: [],
    price_issues: [],
    wave_issues: []
  }
};
```

### 7.2 Metrics Summary
Display in header:
- Total trades: N
- Win rate: XX.X%
- Total return: XX.XX%
- Sharpe ratio: X.XX
- Max drawdown: X.XX%
- Profit factor: X.XX

---

## 8. File Generation

### 8.1 Python Script
Create `backtest/visualization/verification_chart.py`:

```python
def generate_verification_html(
    df: pd.DataFrame,          # OHLC data
    result: BacktestResult,    # From backtest engine
    pivots: List[Dict],        # Pivot data with confirmation
    patterns: List[Dict],      # Wave patterns
    output_path: str           # Where to save HTML
) -> str:
    """
    Generate self-contained HTML verification tool.
    Returns path to generated file.
    """
```

### 8.2 Usage
```python
from backtest.visualization.verification_chart import generate_verification_html
from backtest.engine.backtest_fixed import RealisticBacktestEngine
from backtest.data.storage import DataStorage

storage = DataStorage('data')
df = storage.load('ETHUSDT', '1d')

engine = RealisticBacktestEngine(Config())
result = engine.run(df, 'ETHUSDT', '1d', return_details=True)

html_path = generate_verification_html(
    df=df,
    result=result,
    pivots=result.pivots,
    patterns=result.patterns,
    output_path='output/verification_ETHUSDT_1d.html'
)

print(f"Open {html_path} in browser")
```

---

## 9. Acceptance Criteria

The visualization is complete when:

1. [ ] Candlestick chart displays all ETHUSDT 1D data correctly
2. [ ] Timeline scrubber allows jumping to any point in time
3. [ ] All detected wave patterns are visible (traded and non-traded)
4. [ ] Pivot confirmation arrows clearly show timing
5. [ ] Trade entries/exits display with correct markers
6. [ ] Trade paths connect entry → TP1 → TP2/SL
7. [ ] Fib levels are toggleable (entry zone, TP1, TP2, SL)
8. [ ] Trade info panel shows full accounting detail
9. [ ] Equity curve with drawdown shading syncs with main chart
10. [ ] Trade log table is sortable and click-to-highlight works
11. [ ] Left/right arrow keys navigate between trades
12. [ ] Audit indicators subtly flag any discrepancies
13. [ ] Dark theme renders correctly
14. [ ] File is self-contained, opens in any browser
15. [ ] Real-time MTM updates when scrubbing timeline

---

## 10. Verification Workflow

Expected usage for verification:

1. Open HTML in browser
2. Check metrics summary - do they match CSV output?
3. Use trade log to find first few trades
4. For each trade:
   - Verify entry bar is AFTER confirmed_idx (check arrow)
   - Verify entry price = open of entry bar
   - Verify stop/TP levels match Fib calculations
   - Verify exit prices match OHLC at exit bar
5. Scrub timeline to spot-check random points
6. Toggle failed setups ON to verify non-entries
7. Review any audit flags
8. If all checks pass, backtest is verified correct

---

*Specification Version: 1.0*
*Created: January 6, 2026*
