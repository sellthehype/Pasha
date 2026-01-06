# Elliott Wave Fibonacci Trading System (EWFTS)
## A Comprehensive Rules-Based Trading System for Crypto Markets

---

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Core Principles & Definitions](#2-core-principles--definitions)
3. [Wave Identification Framework](#3-wave-identification-framework)
4. [Module A: Wave 3 Entries](#4-module-a-wave-3-entries)
5. [Module B: Wave 5 Entries](#5-module-b-wave-5-entries)
6. [Module C: Corrective Wave Entries](#6-module-c-corrective-wave-entries)
7. [Stop Loss Framework](#7-stop-loss-framework)
8. [Take Profit & Exit Rules](#8-take-profit--exit-rules)
9. [Position Sizing System](#9-position-sizing-system)
10. [Risk Management Framework](#10-risk-management-framework)
11. [Trade Execution Checklist](#11-trade-execution-checklist)
12. [Module Configuration](#12-module-configuration)
13. [Appendix: Fibonacci Reference Tables](#13-appendix-fibonacci-reference-tables)

---

## 1. System Overview

### 1.1 System Philosophy
This trading system is built entirely on Elliott Wave Theory and Fibonacci mathematics. It exploits the fractal nature of markets where price moves in predictable wave patterns driven by collective market psychology. The system identifies high-probability entry points where Elliott Wave rules provide objective invalidation levels (stops) and Fibonacci relationships provide mathematical profit targets.

### 1.2 System Characteristics
| Attribute | Specification |
|-----------|---------------|
| **Markets** | Cryptocurrency (BTC, ETH, majors, altcoins) |
| **Timeframes** | All (designed timeframe-agnostic) |
| **Style** | Scalping → Position Trading |
| **Base Risk** | 1-2% per trade (adjustable) |
| **Indicators** | None (Pure Elliott Wave + Fibonacci) |
| **Trade Types** | Trend-following (Waves 3, 5) + Mean-reversion (Correctives) |

### 1.3 Modular Design
The system consists of three independent modules that can be activated individually or combined:

| Module | Focus | Risk/Reward Profile |
|--------|-------|---------------------|
| **Module A** | Wave 3 Entries | Highest reward, moderate difficulty |
| **Module B** | Wave 5 Entries | Moderate reward, easier identification |
| **Module C** | Corrective Entries | Lower reward, counter-trend |

---

## 2. Core Principles & Definitions

### 2.1 The Three Inviolable Rules
These rules are ABSOLUTE and form the basis of all stop-loss placement:

| Rule | Statement | Trading Implication |
|------|-----------|---------------------|
| **Rule 1** | Wave 2 NEVER retraces more than 100% of Wave 1 | Stop for Wave 3 entries |
| **Rule 2** | Wave 3 can NEVER be the shortest impulse wave | Minimum W3 target validation |
| **Rule 3** | Wave 4 NEVER enters Wave 1 price territory | Stop for Wave 5 entries |

**Exception to Rule 3**: Diagonal patterns allow Wave 4 to overlap Wave 1.

### 2.2 Wave Definitions

**Motive Waves (Trend Direction)**
- **Impulse**: 5-wave structure moving with the larger trend
  - Waves 1, 3, 5: Move in trend direction (motive)
  - Waves 2, 4: Counter-trend corrections
  - Wave 3: Always the most powerful, never the shortest

- **Diagonal**: 5-wave structure with converging/diverging trendlines
  - Leading Diagonal: Appears in Wave 1 position
  - Ending Diagonal: Appears in Wave 5 position
  - Key feature: Wave 4 overlaps Wave 1

**Corrective Waves (Counter-Trend)**
- **Zigzag (A-B-C)**: Sharp correction, Wave B ≤ 100% of A
- **Flat (A-B-C)**: Sideways correction, Wave B ≥ 90% of A
- **Triangle (A-B-C-D-E)**: Consolidation pattern, converging waves
- **Combinations (W-X-Y or W-X-Y-X-Z)**: Complex corrections

### 2.3 Alternation Principle (Prediction Tool)
| If Wave 2 is... | Then Wave 4 likely... |
|-----------------|----------------------|
| Sharp/Deep (Zigzag) | Sideways/Shallow (Flat/Triangle) |
| Sideways/Shallow | Sharp/Deep |

| If Wave 1 is... | Then expect... |
|-----------------|----------------|
| Short | Wave 3 extended, Wave 5 short |
| Extended | Wave 3 or 5 normal length |

---

## 3. Wave Identification Framework

### 3.1 Confirming You Are in Wave 2 (For Wave 3 Entry)

**Checklist - ALL must be true:**
- [ ] Clear 5-wave or 3-wave impulse completed as Wave 1
- [ ] Current retracement is corrective structure (3-wave or complex)
- [ ] Retracement depth: Between 38.2% and 78.6% of Wave 1
- [ ] Retracement has NOT exceeded 100% of Wave 1
- [ ] Price action is slower/choppier than Wave 1

**High-Probability Wave 2 Fibonacci Zones:**
| Zone | Fibonacci Level | Probability |
|------|-----------------|-------------|
| Primary | 61.8% | Highest |
| Secondary | 50.0% | High |
| Deep | 78.6% | Moderate (sharp corrections) |

### 3.2 Confirming You Are in Wave 4 (For Wave 5 Entry)

**Checklist - ALL must be true:**
- [ ] Waves 1, 2, 3 clearly identifiable
- [ ] Wave 3 is longest OR Wave 1 is shorter than Wave 3
- [ ] Current correction has NOT entered Wave 1 territory
- [ ] Correction structure alternates from Wave 2 type
- [ ] Wave 4 is shallower than Wave 2 (typically)

**High-Probability Wave 4 Fibonacci Zones:**
| Condition | Fibonacci Level | Notes |
|-----------|-----------------|-------|
| Wave 3 Extended | 23.6% - 38.2% | Shallow correction |
| Wave 3 Normal | 38.2% - 50.0% | Moderate correction |

### 3.3 Confirming Corrective Patterns

**Zigzag Identification:**
- [ ] Wave A is impulsive (5-wave structure)
- [ ] Wave B retraces 38% - 79% of Wave A
- [ ] Wave B does NOT exceed Wave A start
- [ ] Wave C targets: 100%, 61.8%, or 161.8% of Wave A

**Flat Identification:**
- [ ] Wave A is corrective (3-wave structure)
- [ ] Wave B retraces ≥ 90% of Wave A
- [ ] Classify variant:
  - Regular: B = 90-105% of A
  - Expanded: B = 105-138% of A (most common)
  - Running: B exceeds A start, C fails to reach A end (rare)

**Triangle Identification:**
- [ ] Five corrective waves (A-B-C-D-E)
- [ ] Wave C does not exceed Wave A
- [ ] Wave D does not exceed Wave B
- [ ] Converging trendlines (contracting) OR diverging (expanding)
- [ ] Appears in Wave 4, Wave B, or final position of combinations

---

## 4. Module A: Wave 3 Entries

### 4.1 Module Overview
Wave 3 is the most powerful wave in Elliott Wave Theory. It cannot be the shortest wave and typically extends to 161.8% of Wave 1. This module captures entries at the end of Wave 2.

**Enabled by default**: YES

### 4.2 Entry Conditions

**LONG Entry - ALL conditions required:**
1. **Trend Context**: Higher timeframe shows bullish bias OR new impulse emerging
2. **Wave 1 Complete**: Clear 5-wave advance OR leading diagonal identified
3. **Wave 2 Structure**: 3-wave or complex corrective structure
4. **Fibonacci Zone**: Price at 50%, 61.8%, or 78.6% retracement of Wave 1
5. **Hold Confirmation**: Price holds above 100% retracement of Wave 1
6. **Momentum Shift**: Price shows reversal candle pattern at Fibonacci level

**SHORT Entry - ALL conditions required:**
1. **Trend Context**: Higher timeframe shows bearish bias OR new impulse emerging
2. **Wave 1 Complete**: Clear 5-wave decline OR leading diagonal identified
3. **Wave 2 Structure**: 3-wave or complex corrective structure
4. **Fibonacci Zone**: Price at 50%, 61.8%, or 78.6% retracement of Wave 1
5. **Hold Confirmation**: Price holds below 100% retracement of Wave 1
6. **Momentum Shift**: Price shows reversal candle pattern at Fibonacci level

### 4.3 Entry Execution

**Primary Entry Method - Limit Order:**
```
Entry Price = Wave 1 End - (Wave 1 Range × Fibonacci Level)

Where Fibonacci Level = 0.618 (primary) | 0.50 (secondary) | 0.786 (aggressive)
```

**Secondary Entry Method - Market Order:**
- Wait for price to reach Fibonacci zone
- Confirm with reversal candle (engulfing, pin bar, etc.)
- Enter on close of confirmation candle

### 4.4 Stop Loss Placement

**Hard Stop (Invalidation):**
```
LONG: Stop = Wave 1 Start - Buffer
SHORT: Stop = Wave 1 Start + Buffer

Buffer = 0.1% to 0.5% of price (accounts for wicks/spread)
```

**Rationale**: If Wave 2 exceeds 100% of Wave 1, the wave count is INVALID.

### 4.5 Take Profit Targets

| Target | Fibonacci Level | Calculation | Priority |
|--------|-----------------|-------------|----------|
| **TP1** | 100% Extension | Wave 1 Range added to Wave 2 End | Primary |
| **TP2** | 161.8% Extension | Wave 1 Range × 1.618 added to Wave 2 End | Primary |
| **TP3** | 261.8% Extension | Wave 1 Range × 2.618 added to Wave 2 End | Extended |

**Minimum Target Requirement:**
Wave 3 MUST exceed Wave 1 length. If TP1 is not reached, reassess wave count.

### 4.6 Example Calculation (LONG)

```
Wave 1 Start: $40,000
Wave 1 End:   $44,000
Wave 1 Range: $4,000

Wave 2 Fibonacci Levels:
- 50.0% = $44,000 - ($4,000 × 0.50) = $42,000
- 61.8% = $44,000 - ($4,000 × 0.618) = $41,528
- 78.6% = $44,000 - ($4,000 × 0.786) = $40,856

Entry (61.8%): $41,528
Stop Loss: $39,950 (below Wave 1 start with buffer)
Risk: $41,528 - $39,950 = $1,578

Targets from Wave 2 end at $41,528:
- TP1 (100%): $41,528 + $4,000 = $45,528
- TP2 (161.8%): $41,528 + $6,472 = $48,000
- TP3 (261.8%): $41,528 + $10,472 = $52,000

Risk:Reward Ratios:
- TP1: 1:2.53
- TP2: 1:4.10
- TP3: 1:6.63
```

---

## 5. Module B: Wave 5 Entries

### 5.1 Module Overview
Wave 5 is the final wave of an impulse sequence. It offers easier identification since Waves 1-4 are already visible. However, it carries truncation risk (Wave 5 failing to exceed Wave 3).

**Enabled by default**: YES

### 5.2 Entry Conditions

**LONG Entry - ALL conditions required:**
1. **Waves 1-4 Visible**: Clear impulse structure with W1, W2, W3, W4 identifiable
2. **Wave 3 Valid**: Wave 3 is not the shortest impulse wave
3. **Wave 4 Structure**: Corrective pattern alternating from Wave 2
4. **No Overlap**: Wave 4 has NOT entered Wave 1 price territory
5. **Fibonacci Zone**: Price at 23.6%, 38.2%, or 50% retracement of Wave 3
6. **Momentum Shift**: Price shows reversal at support

**SHORT Entry - ALL conditions required:**
1. **Waves 1-4 Visible**: Clear impulse structure with W1, W2, W3, W4 identifiable
2. **Wave 3 Valid**: Wave 3 is not the shortest impulse wave
3. **Wave 4 Structure**: Corrective pattern alternating from Wave 2
4. **No Overlap**: Wave 4 has NOT entered Wave 1 price territory
5. **Fibonacci Zone**: Price at 23.6%, 38.2%, or 50% retracement of Wave 3
6. **Momentum Shift**: Price shows reversal at resistance

### 5.3 Entry Execution

**Primary Entry Method - Limit Order:**
```
LONG Entry = Wave 3 End - (Wave 3 Range × Fibonacci Level)
SHORT Entry = Wave 3 End + (Wave 3 Range × Fibonacci Level)

Fibonacci Level = 0.382 (primary) | 0.236 (shallow W3 extended) | 0.50 (deep)
```

### 5.4 Stop Loss Placement

**Hard Stop (Invalidation):**
```
LONG: Stop = Wave 1 High - Buffer (overlap invalidation)
SHORT: Stop = Wave 1 Low + Buffer

Alternative (Tighter):
LONG: Stop = Wave 4 Low - Buffer
SHORT: Stop = Wave 4 High + Buffer
```

**Use Wave 1 level for invalidation-based stop, Wave 4 level for tighter risk management.**

### 5.5 Take Profit Targets

**Wave 5 typically shows one of these relationships to Wave 1:**

| Target | Relationship | Calculation |
|--------|--------------|-------------|
| **TP1** | W5 = W1 | Wave 1 Range added to Wave 4 End |
| **TP2** | W5 = 61.8% of W1 | Wave 1 Range × 0.618 added to Wave 4 End |
| **TP3** | W5 = 161.8% of W1 | Wave 1 Range × 1.618 added to Wave 4 End |

**If Wave 3 is Extended:**
Waves 1 and 5 tend toward equality OR 61.8% relationship.

**Target Selection Logic:**
```
IF Wave 3 > 161.8% of Wave 1:
    Primary Target = Wave 5 equals Wave 1 (TP1)
ELSE:
    Primary Target = Wave 5 at 161.8% extension (TP3)
```

### 5.6 Truncation Risk Management

A truncation occurs when Wave 5 fails to exceed Wave 3's end.

**Warning Signs:**
- Wave 3 was extremely extended (> 261.8% of Wave 1)
- Divergence between price and momentum (on higher timeframe)
- Low volume on Wave 5 advance

**Mitigation:**
- Take 50% profit at TP1
- Move stop to breakeven once TP1 reached
- Reduce position size for Wave 5 trades vs Wave 3 trades

---

## 6. Module C: Corrective Wave Entries

### 6.1 Module Overview
Corrective wave entries are counter-trend trades within the larger wave structure. They capture moves within Waves 2, 4, or larger corrections. Higher risk but can be profitable in ranging markets.

**Enabled by default**: YES (can be disabled)

### 6.2 Zigzag Entries

#### 6.2.1 Wave C Entry (Primary)
Enter at completion of Wave B to ride Wave C.

**Entry Conditions:**
1. Wave A is impulsive (5-wave structure identified)
2. Wave B is corrective and retraces 38% - 79% of Wave A
3. Wave B does NOT exceed Wave A start (invalidation)
4. Entry at Fibonacci level of Wave A retracement

**Entry Zone:**
```
Entry = Wave A End + (Wave A Range × B Retracement)
B Retracement Zones: 38.2%, 50%, 61.8%, 78.6%
```

**Stop Loss:**
```
Stop = Wave A Start + Buffer (B exceeding A start = invalid)
```

**Take Profit Targets:**
| Target | Wave C Relationship | Calculation |
|--------|---------------------|-------------|
| **TP1** | C = 61.8% of A | Wave A Range × 0.618 from B End |
| **TP2** | C = 100% of A | Wave A Range × 1.00 from B End |
| **TP3** | C = 161.8% of A | Wave A Range × 1.618 from B End |

### 6.3 Flat Entries

#### 6.3.1 Expanded Flat Wave C Entry
The expanded flat is the MOST COMMON flat variant.

**Entry Conditions:**
1. Wave A is corrective (3-wave structure)
2. Wave B retraces 105% - 138% of Wave A (exceeds A start)
3. Wave B complete (3-wave corrective structure)

**Entry:**
```
Enter at completion of Wave B (when 3-wave B structure complete)
```

**Stop Loss:**
```
Stop = Wave B End + Buffer × ATR(20)
(Beyond Wave B extreme since C must move significantly)
```

**Take Profit Targets:**
| Target | Wave C Relationship | Notes |
|--------|---------------------|-------|
| **TP1** | C = 100% of A | Minimum expectation |
| **TP2** | C = 161.8% of A | Common target |
| **TP3** | C = 261.8% of A | Rare but possible |

**Note:** Wave C in expanded flats typically moves well beyond Wave A's end.

### 6.4 Triangle Entries

#### 6.4.1 Post-Triangle Thrust Entry
Triangles are consolidation patterns. The thrust after completion provides a high-probability entry.

**Entry Conditions:**
1. Clear 5-wave triangle structure (A-B-C-D-E) identified
2. Wave C did not exceed Wave A
3. Wave D did not exceed Wave B
4. Converging trendlines with at least 4 touch points
5. Wave E complete (touched or nearly touched lower trendline)

**Entry:**
```
Enter on breakout from triangle (Wave E completion)
Direction = Same as wave BEFORE the triangle
```

**Stop Loss:**
```
Stop = Wave E extreme + Buffer
(Triangle should not be re-entered after thrust begins)
```

**Take Profit:**
```
Thrust Target = Triangle Width at Start (Wave A to B range)

Measurement:
1. Draw vertical line at start of triangle (Wave A)
2. Measure distance between upper and lower trendlines
3. Project that distance from triangle breakout point
```

---

## 7. Stop Loss Framework

### 7.1 Stop Loss Philosophy
All stops in this system are based on **Elliott Wave INVALIDATION** levels. When a stop is hit, the wave count was wrong—this is objective, not emotional.

### 7.2 Stop Loss Rules by Module

| Module | Trade Type | Stop Location | Invalidation Reason |
|--------|------------|---------------|---------------------|
| **A** | Wave 3 Entry | Below/Above Wave 1 Start | W2 > 100% invalidates count |
| **B** | Wave 5 Entry (Wide) | Below/Above Wave 1 End | W4 overlap invalidates impulse |
| **B** | Wave 5 Entry (Tight) | Below/Above Wave 4 Extreme | Practical risk management |
| **C** | Zigzag C Entry | Beyond Wave A Start | B exceeding A invalidates zigzag |
| **C** | Flat C Entry | Beyond Wave B Extreme | Pattern failure |
| **C** | Triangle Thrust | Beyond Wave E Extreme | Triangle failure |

### 7.3 Buffer Calculation

```
Buffer = MAX(0.1% of Entry Price, 0.5 × ATR(20))

Purpose: Account for wicks, spread, and market noise
```

### 7.4 Stop Loss Adjustment Rules

**Initial Stop**: NEVER move stop AGAINST the trade. The invalidation level is fixed.

**Breakeven Stop**: Move stop to entry when:
- Price reaches 1:1 Risk:Reward
- First take profit target hit
- Pattern confirmation strengthens

**Trailing Stop (Optional):**
Not explicitly required by Elliott Wave theory but can be implemented:
```
Trailing Stop = Recent Swing Low/High - Buffer

Update when:
- New higher low (longs) or lower high (shorts) forms
- Sub-wave completes within the trade
```

---

## 8. Take Profit & Exit Rules

### 8.1 Exit Philosophy
Exits are determined by Fibonacci extensions derived from Elliott Wave relationships. The market will often respect these mathematical levels.

### 8.2 Primary Exit Strategy - Multiple Targets

**Position Distribution:**
| Portion | Target | Action |
|---------|--------|--------|
| 40% | TP1 | Close at first target |
| 40% | TP2 | Close at second target |
| 20% | TP3 | Close at extended target OR trail |

### 8.3 Exit Targets by Module

#### Module A (Wave 3)
| Target | Calculation | R:R Expectation |
|--------|-------------|-----------------|
| TP1 | 100% extension of W1 from W2 end | 1:2 - 1:3 |
| TP2 | 161.8% extension of W1 from W2 end | 1:4 - 1:5 |
| TP3 | 261.8% extension of W1 from W2 end | 1:6+ |

#### Module B (Wave 5)
| Target | Calculation | Notes |
|--------|-------------|-------|
| TP1 | W5 = 61.8% of W1 | Conservative |
| TP2 | W5 = 100% of W1 | Most common if W3 extended |
| TP3 | W5 = 161.8% of W1 | If W3 not extended |

#### Module C (Correctives)
| Pattern | TP1 | TP2 | TP3 |
|---------|-----|-----|-----|
| Zigzag | C = 61.8% of A | C = 100% of A | C = 161.8% of A |
| Flat | C = 100% of A | C = 161.8% of A | C = 261.8% of A |
| Triangle Thrust | 50% of width | 100% of width | — |

### 8.4 Early Exit Conditions

Exit entire position immediately if:
1. **Wave Count Invalidation**: Price action suggests different wave structure
2. **Time Violation**: Corrective wave takes longer than impulse wave it's correcting
3. **Opposite Setup Triggers**: Valid entry signal in opposite direction appears
4. **Sub-Wave 5 of 3 Complete**: When trading W3, exit if clear 5-wave structure within W3 completes

---

## 9. Position Sizing System

### 9.1 Core Position Sizing Formula

```
Position Size = Risk Amount ÷ Risk Per Unit

Where:
- Risk Amount = Account Balance × Risk Percentage
- Risk Per Unit = |Entry Price - Stop Loss Price|
```

### 9.2 Risk Percentage Settings

**Fixed Risk Model:**
| Setting | Risk % | Use Case |
|---------|--------|----------|
| Conservative | 0.5% | Uncertain wave count, learning phase |
| Moderate | 1.0% | Standard trading |
| Moderate-Plus | 1.5% | Clear wave structure |
| Aggressive | 2.0% | High-conviction setups |

**Adjustable Parameter:**
```
RISK_PERCENTAGE = 1.0  // Default: 1%, Range: 0.5% - 2.0%
```

### 9.3 ATR-Based Dynamic Position Sizing

Uses 20-period ATR to adjust for volatility.

**Formula:**
```
ATR_20 = 20-period Average True Range
Baseline_ATR = Historical average ATR for the asset

Volatility_Ratio = ATR_20 ÷ Baseline_ATR

Adjusted_Risk = Base_Risk_Percentage ÷ Volatility_Ratio

IF Volatility_Ratio > 1.5:
    Adjusted_Risk = Base_Risk_Percentage ÷ 1.5  // Cap reduction
IF Volatility_Ratio < 0.5:
    Adjusted_Risk = Base_Risk_Percentage ÷ 0.5  // Cap increase
```

**Example:**
```
Account: $100,000
Base Risk: 1% ($1,000)
Current ATR(20): $1,500
Baseline ATR: $1,000
Volatility Ratio: 1.5

Adjusted Risk = 1% ÷ 1.5 = 0.67% ($670)

Entry: $50,000
Stop: $48,500
Risk per unit: $1,500

Position Size = $670 ÷ $1,500 = 0.447 BTC
```

### 9.4 ATR-Based Stop Distance Alternative

Instead of adjusting risk percentage, use ATR to set stop distance:

```
Stop Distance = ATR(20) × Multiplier

Multiplier by Module:
- Module A (Wave 3): 2.0 × ATR
- Module B (Wave 5): 1.5 × ATR
- Module C (Corrective): 1.0 × ATR

Final Stop = MAX(ATR Stop, Invalidation Stop)
```

### 9.5 Position Sizing by Module

| Module | Base Risk | Confidence Multiplier |
|--------|-----------|----------------------|
| **A** (Wave 3) | 1.0× | 1.0 - 1.25× (higher confidence) |
| **B** (Wave 5) | 1.0× | 0.75 - 1.0× (truncation risk) |
| **C** (Corrective) | 1.0× | 0.5 - 0.75× (counter-trend) |

**Confidence Multiplier Application:**
```
Final_Risk = Base_Risk × Module_Multiplier × Confidence_Factor

Confidence_Factor based on:
- Wave count clarity (0.8 - 1.2)
- Fibonacci level precision (0.9 - 1.1)
- Multiple timeframe alignment (0.9 - 1.2)
```

---

## 10. Risk Management Framework

### 10.1 Per-Trade Risk Rules

| Rule | Limit | Notes |
|------|-------|-------|
| Maximum risk per trade | 2% of account | Hard ceiling |
| Standard risk per trade | 1% of account | Default |
| Minimum risk per trade | 0.25% of account | Floor for micro-managing |

### 10.2 Portfolio Heat (Total Exposure)

```
Maximum Portfolio Heat = 6% of Account

Portfolio Heat = Sum of all open trade risks
```

**Implementation:**
- Maximum 6 trades at 1% risk each
- Maximum 3 trades at 2% risk each
- Reduce new position sizes if heat approaches limit

### 10.3 Correlation Rules

**Same Asset:**
- Maximum 2 positions in same asset across timeframes
- Combined risk ≤ 3% if correlated direction

**Correlated Assets:**
- BTC and ETH considered correlated
- Alt/BTC pairs correlated with BTC/USD
- Reduce aggregate exposure when trading correlated pairs

### 10.4 Drawdown Controls

| Drawdown Level | Action |
|----------------|--------|
| 5% | Review: Assess all open positions |
| 10% | Reduce: Cut position size by 50% |
| 15% | Pause: Close all trades, analyze errors |
| 20% | Stop: No trading until full system review |

### 10.5 Win Rate Expectations

**Expected Performance by Module:**
| Module | Expected Win Rate | Avg R:R | Expectancy |
|--------|-------------------|---------|------------|
| **A** | 45-55% | 1:3 | 0.35-0.65 |
| **B** | 40-50% | 1:2 | 0.20-0.50 |
| **C** | 35-45% | 1:1.5 | 0.02-0.18 |

**Combined System Target:**
- Win Rate: 40-50%
- Average R:R: 1:2.5
- Expectancy: 0.20+ per trade

---

## 11. Trade Execution Checklist

### 11.1 Pre-Trade Checklist (All Trades)

```
□ STEP 1: WAVE COUNT VERIFICATION
  □ Can I identify clear wave structure?
  □ Do the three inviolable rules hold?
  □ Does alternation principle align?
  □ Is the wave count the SIMPLEST explanation?

□ STEP 2: MODULE SELECTION
  □ Which module does this setup belong to?
  □ Is that module currently enabled?
  □ What is the confidence level (High/Medium/Low)?

□ STEP 3: ENTRY VALIDATION
  □ Is price at a valid Fibonacci level?
  □ Are all entry conditions for the module met?
  □ Is there a reversal signal/pattern at entry zone?

□ STEP 4: STOP LOSS CALCULATION
  □ Where is the invalidation level?
  □ What is the stop with buffer?
  □ Is the risk per unit acceptable?

□ STEP 5: TAKE PROFIT TARGETS
  □ What are TP1, TP2, TP3 levels?
  □ Is minimum R:R of 1:2 achievable to TP1?
  □ Are targets at logical Fibonacci extensions?

□ STEP 6: POSITION SIZING
  □ What is current portfolio heat?
  □ What risk percentage applies?
  □ Is ATR adjustment needed?
  □ Final position size calculated?

□ STEP 7: EXECUTION
  □ Limit order placed at entry?
  □ Stop loss order placed?
  □ Take profit orders placed?
  □ Trade logged in journal?
```

### 11.2 Module A (Wave 3) Specific Checklist

```
□ Wave 1 is a clear 5-wave impulse OR leading diagonal
□ Wave 2 is a 3-wave OR complex correction
□ Wave 2 has retraced 38.2% - 78.6% of Wave 1
□ Wave 2 has NOT exceeded 100% of Wave 1
□ Entry is at 50%, 61.8%, or 78.6% Fibonacci level
□ Stop is below/above Wave 1 start
□ TP2 targets 161.8% extension of Wave 1
```

### 11.3 Module B (Wave 5) Specific Checklist

```
□ Waves 1, 2, 3, 4 clearly visible
□ Wave 3 is NOT the shortest wave
□ Wave 4 has NOT entered Wave 1 territory
□ Wave 4 pattern alternates from Wave 2
□ Entry is at 23.6%, 38.2%, or 50% of Wave 3
□ Stop is below/above Wave 1 end (or Wave 4 extreme for tight stop)
□ TP1 considers W1-W5 equality if W3 was extended
□ Truncation risk has been assessed
```

### 11.4 Module C (Corrective) Specific Checklist

```
□ Larger degree trend direction identified
□ Corrective pattern type identified (Zigzag/Flat/Triangle)
□ Pattern-specific rules verified
□ Entry at appropriate Fibonacci level
□ Stop at pattern invalidation level
□ Target based on pattern-specific projections
□ Counter-trend risk accepted (reduced position size)
```

---

## 12. Module Configuration

### 12.1 Module Enable/Disable Settings

```
MODULES_CONFIG = {
    "MODULE_A_WAVE3": true,    // Wave 3 entries
    "MODULE_B_WAVE5": true,    // Wave 5 entries
    "MODULE_C_CORRECTIVE": true // Corrective entries
}

// Sub-configurations for Module C
MODULE_C_CONFIG = {
    "ZIGZAG_ENTRIES": true,
    "FLAT_ENTRIES": true,
    "TRIANGLE_ENTRIES": true,
    "COMBINATION_ENTRIES": false  // Advanced, disabled by default
}
```

### 12.2 Risk Parameter Configuration

```
RISK_CONFIG = {
    // Base risk percentage
    "BASE_RISK_PERCENT": 1.0,       // Range: 0.5 - 2.0

    // Dynamic ATR sizing
    "USE_ATR_SIZING": true,         // Enable/disable ATR adjustment
    "ATR_PERIOD": 20,               // ATR lookback period
    "ATR_MULTIPLIER_CAP": 1.5,      // Maximum adjustment

    // Portfolio limits
    "MAX_PORTFOLIO_HEAT": 6.0,      // Maximum total exposure %
    "MAX_TRADES_SAME_ASSET": 2,     // Correlation limit
    "MAX_CORRELATED_EXPOSURE": 4.0  // For BTC/ETH together
}
```

### 12.3 Module-Specific Risk Multipliers

```
MODULE_RISK_MULTIPLIERS = {
    "MODULE_A_WAVE3": 1.0,          // Full risk
    "MODULE_B_WAVE5": 0.75,         // Reduced for truncation risk
    "MODULE_C_CORRECTIVE": 0.5     // Counter-trend, half risk
}
```

### 12.4 Timeframe Configuration

```
TIMEFRAME_CONFIG = {
    // Enable trading on these timeframes
    "1m": true,     // Scalping
    "5m": true,     // Scalping
    "15m": true,    // Day trading
    "1h": true,     // Day/Swing
    "4h": true,     // Swing
    "1d": true,     // Position
    "1w": true,     // Position

    // Higher timeframe for trend context
    "CONTEXT_TF_MULTIPLIER": 4  // Use 4× timeframe for trend
}
```

---

## 13. Appendix: Fibonacci Reference Tables

### 13.1 Retracement Levels

| Level | Decimal | Use Case |
|-------|---------|----------|
| 23.6% | 0.236 | Shallow Wave 4 (W3 extended) |
| 38.2% | 0.382 | Standard Wave 4, minimum Zigzag B |
| 50.0% | 0.500 | Wave 2 primary, Wave 4 moderate |
| 61.8% | 0.618 | Wave 2 golden ratio, Zigzag B |
| 78.6% | 0.786 | Deep Wave 2, maximum Zigzag B |
| 88.6% | 0.886 | Flat B minimum threshold |
| 100% | 1.000 | Wave 2 maximum, Zigzag B maximum |

### 13.2 Extension Levels

| Level | Decimal | Use Case |
|-------|---------|----------|
| 61.8% | 0.618 | Wave 5 minimum if W3 extended |
| 100% | 1.000 | Wave equality, Zigzag C = A |
| 127.2% | 1.272 | Intermediate extension |
| 138.2% | 1.382 | Flat B maximum |
| 161.8% | 1.618 | Wave 3 typical, Wave C extended |
| 200% | 2.000 | Strong extensions |
| 261.8% | 2.618 | Wave 3 extreme, rare Flat C |
| 361.8% | 3.618 | Exceptional extensions |
| 423.6% | 4.236 | Rare, blow-off moves |

### 13.3 Wave Relationship Quick Reference

**Impulse Waves:**
```
Wave 2: Retraces 50% - 78.6% of Wave 1 (never > 100%)
Wave 3: Typically 161.8% of Wave 1 (never shortest)
Wave 4: Retraces 23.6% - 50% of Wave 3 (no W1 overlap)
Wave 5: Equals Wave 1, or 61.8%, or 161.8% of Wave 1
```

**Corrective Waves:**
```
Zigzag:
- Wave B: 38% - 79% of Wave A
- Wave C: 61.8%, 100%, or 161.8% of Wave A

Flat:
- Wave B: 90% - 138% of Wave A
- Wave C: 100% - 161.8% of Wave A (261.8% rare)

Triangle:
- Each wave: 61.8% - 78.6% of previous wave
- Post-thrust: Equals width of triangle at start
```

---

## Document Information

**System Name:** Elliott Wave Fibonacci Trading System (EWFTS)
**Version:** 1.0
**Created:** Based on Elite CurrenSea Elliott Wave Reference Guide
**Markets:** Cryptocurrency
**Approach:** Pure Elliott Wave + Fibonacci (No additional indicators)

---

*This trading system is for educational purposes. Always practice proper risk management and never risk more than you can afford to lose.*
