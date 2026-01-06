# Elliott Wave Trading Strategy
## The Simple Guide

---

## What Is This Strategy?

This strategy catches big price moves by identifying **where we are in a wave cycle**. Markets move in patterns: up in 5 waves, down in 3 waves (or vice versa). We enter trades when one wave ends and another begins.

**The Edge:** Elliott Wave tells us exactly where our trade is WRONG, giving us precise stop losses. Fibonacci math tells us where price is likely to go, giving us profit targets.

---

## The 3 Golden Rules (Never Broken)

| Rule | What It Means | Why You Care |
|------|---------------|--------------|
| **Rule 1** | Wave 2 can't erase Wave 1 completely | Your Wave 3 stop goes just below Wave 1 start |
| **Rule 2** | Wave 3 is never the smallest wave | Wave 3 trades have the biggest targets |
| **Rule 3** | Wave 4 can't crash into Wave 1's territory | Your Wave 5 stop goes just above Wave 1's high |

If any rule breaks → your wave count was wrong → stop loss was correct → move on.

---

## The 3 Trade Types

### Trade Type A: Catching Wave 3 (The Big One)

**The Setup:**
```
Price made a clear move up (Wave 1)
Then pulled back but didn't erase the move (Wave 2)
Now sitting at a Fibonacci level, ready to explode higher (Wave 3)
```

**When to Enter:**
- Price pulled back to 50%, 61.8%, or 78.6% of Wave 1
- Shows signs of reversing (bounce, bullish candle, etc.)
- Has NOT dropped below where Wave 1 started

**Where to Put Stop:**
- Just below Wave 1 starting point
- Add a small buffer (0.1-0.5%) for safety

**Where to Take Profit:**
- Target 1: Wave 1 length added to your entry → Exit 40%
- Target 2: Wave 1 × 1.618 added to your entry → Exit 40%
- Target 3: Wave 1 × 2.618 added to your entry → Exit final 20%

**Why This Trade Rocks:**
Wave 3 is the most powerful wave. Often gives 3:1 to 6:1 reward-to-risk.

---

### Trade Type B: Catching Wave 5 (The Final Push)

**The Setup:**
```
You can clearly see Waves 1, 2, 3, and 4
Wave 4 pulled back but stayed above Wave 1's high
One more push up expected (Wave 5)
```

**When to Enter:**
- Price pulled back 23.6% to 38.2% of Wave 3
- Wave 4 is a DIFFERENT pattern than Wave 2 was
- Price shows reversal signs

**Where to Put Stop:**
- Safe stop: Below Wave 1's high point
- Tight stop: Below Wave 4's low point

**Where to Take Profit:**
- If Wave 3 was huge: Target Wave 5 = Wave 1 length
- If Wave 3 was normal: Target Wave 5 = Wave 1 × 1.618

**The Risk:**
Wave 5 can "truncate" (fail to make new highs). Use smaller position sizes than Wave 3 trades.

---

### Trade Type C: Catching Correction Moves

**Zigzag (Sharp Pullback):**
```
Price dropped sharply in Wave A
Bounced back 38-79% in Wave B
Enter for Wave C down (or vice versa for upward zigzag)
```
- Stop: Beyond Wave A's start
- Target: Wave C = Wave A length (or 61.8% or 161.8%)

**Triangle Breakout:**
```
Price consolidating in shrinking range (A-B-C-D-E)
Each wave smaller than the last
Enter when Wave E completes
```
- Stop: Beyond Wave E extreme
- Target: Width of triangle at start

**Flat (Sideways Pullback):**
```
Wave A was weak/corrective
Wave B retraced almost all (or more) of A
Enter for Wave C
```
- Target: C often overshoots A's end significantly

---

## Stop Loss Cheat Sheet

| Trade Type | Stop Location | Logic |
|------------|---------------|-------|
| Wave 3 Entry | Below Wave 1 START | If W2 > 100%, count is wrong |
| Wave 5 Entry | Below Wave 1 HIGH | If W4 overlaps W1, count is wrong |
| Wave 5 (Tight) | Below Wave 4 LOW | Tighter risk, same trade |
| Zigzag C | Beyond Wave A START | If B > A, it's not a zigzag |
| Triangle | Beyond Wave E | If triangle fails, pattern wrong |

**Buffer Rule:**
```
Always add buffer = MAX(0.1% of price, half of ATR)
This protects against wicks and spread
```

---

## Take Profit Cheat Sheet

### Wave 3 Targets (from your entry point)
| Target | Calculation | Exit |
|--------|-------------|------|
| TP1 | Entry + (Wave 1 size) | 40% |
| TP2 | Entry + (Wave 1 × 1.618) | 40% |
| TP3 | Entry + (Wave 1 × 2.618) | 20% |

### Wave 5 Targets
| Condition | Target |
|-----------|--------|
| Wave 3 was extended (>161.8%) | Wave 5 = Wave 1 |
| Wave 3 was normal | Wave 5 = Wave 1 × 1.618 |

### Correction Targets
| Pattern | Target |
|---------|--------|
| Zigzag C | = Wave A (or 61.8% or 161.8% of A) |
| Flat C | = 100-161.8% of Wave A |
| Triangle Thrust | = Width of triangle |

---

## Position Sizing Made Simple

### The Basic Formula
```
How much to buy = Money you're risking ÷ Distance to stop loss

Example:
- Account: $10,000
- Risk: 1% = $100
- Entry: $50,000
- Stop: $49,000
- Distance: $1,000

Position = $100 ÷ $1,000 = 0.1 BTC
```

### Risk Levels
| Confidence | Risk % | When to Use |
|------------|--------|-------------|
| Low | 0.5% | Unclear wave count |
| Normal | 1.0% | Standard setup |
| High | 1.5% | Perfect wave structure |
| Maximum | 2.0% | Only the best setups |

### Adjusting for Volatility (ATR Method)
```
If market is CRAZY volatile (ATR much higher than normal):
→ Reduce your risk % proportionally

If market is CALM (ATR lower than normal):
→ You can increase risk % slightly

Formula:
Adjusted Risk = Base Risk ÷ (Current ATR ÷ Normal ATR)

Cap adjustments between 0.5x and 1.5x
```

### By Trade Type
| Trade | Risk Multiplier | Reason |
|-------|-----------------|--------|
| Wave 3 | 100% | Best risk/reward |
| Wave 5 | 75% | Truncation risk |
| Corrections | 50% | Counter-trend |

---

## Portfolio Rules

### Maximum Heat
```
Never risk more than 6% of account across ALL open trades

Examples:
- 6 trades at 1% each = OK
- 3 trades at 2% each = OK
- 4 trades at 2% each = TOO MUCH
```

### Correlation Warning
BTC and ETH often move together. If you have:
- 2% on BTC long
- 2% on ETH long

You really have 4% exposed to "crypto goes up." Treat them as one position.

### Drawdown Rules
| Account Down | Action |
|--------------|--------|
| 5% | Pause and review |
| 10% | Cut position sizes by 50% |
| 15% | Stop trading, analyze what's wrong |
| 20% | Full system review before continuing |

---

## Quick Entry Checklist

Before EVERY trade, confirm:

```
□ Can I clearly count the waves?
□ Do all 3 golden rules hold?
□ Is price at a Fibonacci level?
□ Do I see a reversal signal?
□ Do I know EXACTLY where my stop goes?
□ Is my reward at least 2x my risk?
□ Is my total portfolio risk under 6%?
□ Did I calculate position size correctly?
```

If any box is unchecked → NO TRADE

---

## Entry Signals at Fibonacci Levels

What counts as a "reversal signal" at your entry zone:

**Strong Signals:**
- Engulfing candle (big candle swallows previous)
- Pin bar / hammer (long wick rejection)
- Morning/evening star pattern
- Double bottom at Fibonacci level

**Moderate Signals:**
- Doji at support/resistance
- Decreasing momentum into the level
- Volume spike with rejection

**Weak (Wait for More):**
- Just touching the level
- No candle confirmation
- Still making new lows/highs

---

## Common Mistakes to Avoid

### 1. Forcing Wave Counts
If you have to "make it fit" → it probably doesn't. The best counts are obvious.

### 2. Moving Stops
Your stop is at the invalidation level for a reason. If it hits, you were WRONG. Don't move it further away.

### 3. Oversizing on Wave 5
Wave 5 can fail (truncate). It's the last wave before reversal. Always use smaller size than Wave 3.

### 4. Trading Every Pattern
Not every wave structure is tradeable. Wait for CLEAR setups at GOOD Fibonacci levels.

### 5. Ignoring the Larger Picture
A Wave 3 on the 5-minute chart might be noise in a larger Wave 4 on the daily. Always check higher timeframes.

### 6. Taking Profits Too Early
Wave 3 often goes to 161.8% or beyond. Don't exit everything at the first target. Use the 40/40/20 split.

---

## The 5-Minute Version

**When to Buy (Long):**
1. Market made a move up (Wave 1)
2. Pulled back 50-78% without erasing it (Wave 2)
3. Showing signs of bouncing at Fibonacci level
4. Buy with stop below Wave 1 start

**When to Sell (Short):**
1. Market made a move down (Wave 1)
2. Bounced 50-78% without erasing it (Wave 2)
3. Showing signs of rejection at Fibonacci level
4. Sell with stop above Wave 1 start

**Where to Exit:**
- First target: 100% extension (exit 40%)
- Second target: 161.8% extension (exit 40%)
- Third target: 261.8% extension (exit 20%)

**How Much to Trade:**
- Risk 1% of account per trade
- Never more than 6% total exposure
- Reduce size for Wave 5 and correction trades

---

## Fibonacci Levels Quick Reference

### Pullback Levels (Where to Enter)
| Level | Best For |
|-------|----------|
| 38.2% | Shallow Wave 4 |
| 50.0% | Wave 2, Wave 4 |
| 61.8% | Wave 2 (golden ratio) |
| 78.6% | Deep Wave 2 |

### Extension Levels (Where to Exit)
| Level | Best For |
|-------|----------|
| 100% | Wave 5 = Wave 1, TP1 for Wave 3 |
| 161.8% | Wave 3 typical target |
| 261.8% | Extended Wave 3 |

---

## Final Thoughts

This strategy works because:
1. **Clear invalidation** = You always know when you're wrong
2. **Mathematical targets** = Removes guesswork from exits
3. **Wave psychology** = Markets really do move this way

The hard part is **patience**. Wait for the setup. If it's not clear, don't trade.

Good luck, and remember: the best trade is often no trade at all.

---

*File: Elliott_Wave_Strategy_Summary.md*
*Companion to: Elliott_Wave_Trading_System.md*
