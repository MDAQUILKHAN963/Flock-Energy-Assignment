# Recommendation: estimating washing-machine use from smart-meter data

*For: utility programme, product and network teams. Evidence: 14 UK homes from the REFIT study,
2013–2015; full details in RESULTS.md.*

## What we built

A model that reads a home's **whole-house electricity, minute by minute**, and estimates when the
washing machine is running and how much electricity it uses. No device is fitted in the home. It was
tested on **three homes it had never seen**, the way it would be used on new customers.

| On unseen homes, the model… | Result |
|---|---|
| estimates total washing electricity over the year within | **15 %** (it under-estimates) |
| estimates each day's washing electricity with an average error of | **0.28 kWh/day** (about 40 % better than simple baselines) |
| finds this share of actual washes | **80 %** |
| is right this often when it says a wash happened | **53 %** |

Washing machines use **1–7 % of a home's electricity** (0.7–6 kWh per week), at about 0.5 kWh per wash.

## Suitable uses

- **Portfolio and segment insight:** how much electricity goes to laundry, at what times of day,
  across thousands of homes. Errors largely average out over many homes and weeks.
- **Finding households with a pattern**, e.g. those who regularly wash in the evening peak. Judge
  this over **several weeks** of data, never from one day.
- **Personalised advice that is framed as an estimate** ("it looks like you often wash around 6 pm").

## Not suitable

- **Billing or any per-wash claim.** About half of the washes it detects are not washes. They are
  often a **dishwasher, tumble dryer** or other heating appliance with the same "2 kW heater + motor"
  pattern.
- **Real-time control** (e.g. switching a machine): the model looks an hour either side of each minute.
- **Homes with solar panels, or with only 15–30-minute meter data**, without re-testing (below).
- **Cold/eco washes:** the model misses more of these, so it under-counts households that wash cool.

---

## Section 4 answers

### 1. One practical opportunity: an evening-peak washing nudge

**Insight.** 18 % of weekday washes start between **16:00 and 20:00**, when the network peaks, and
four of fourteen homes do **31–53 %** of their weekday washing then. Washing machines add about
**25 W per household at 19:00** on average, roughly **2.5 MW per 100,000 homes**, mostly from the
2 kW heater at the start of each wash. Two homes in the study already wash mostly at night or late
in the evening (77 % and 58 % of their washes), so moving the wash is clearly practical for some
households.

**Offer.** Use the model to identify customers who wash in the evening peak **repeatedly over
4+ weeks**. Send them a short message: "use your machine's delay-start to finish by morning". Pair it
with a time-of-use tariff where one exists. This saves the customer money and reduces peak load.
Overall consumption barely changes.

**How to measure it.** Run it as a **randomised trial**. Split the eligible customers at random into
a message group and a no-message control group.
- *Primary outcome, measured directly from the meter, not the model:* average household demand
  16:00–20:00 on weekdays, compared before and after between the two groups (difference-in-
  differences over 8–12 weeks).
- *Secondary:* the model's estimate of the share of washing energy in the peak window. Its errors
  are the same in both groups, so the comparison stays fair.
- *Check:* fit smart plugs to ~50 trial machines to confirm what the model sees.

Success means a statistically significant drop in peak-window demand in the message group, which
persists after the first month. A caveat on targeting: dishwashers often run after dinner, so some
"evening washers" will be dishwasher users. Requiring a consistent multi-week pattern reduces this.

### 2. Limitations

- **Overlapping signatures.** Any appliance with a ~2 kW heater and a motor looks like a wash. In
  the test homes the **dishwasher caused 28–46 %** and the **tumble dryer 31 %** of false-alarm
  energy. Un-metered loads (immersion heater, oven, shower) caused most of the rest. A model
  trained specifically to tell these appliances apart improved results only slightly. At 1-minute
  resolution the whole-house signal simply does not carry enough information to separate them
  reliably.
- **Missing data.** 6–25 % of minutes are missing per home. The long gaps hit **every home at once**
  (central system failures). The official data also contains frozen readings and zero-filled gaps,
  which had to be detected and removed. Real smart-meter feeds will have gaps too, so outputs must
  say how much data they rest on.
- **Household differences.** Accuracy varies a lot between homes (the model's detection score ranged
  from 0.33 to 0.72). Energy per wash varies 3× (cool vs hot washes), and each home has its own
  routine. Results are reliable for groups of homes, much less so for any single home.
- **UK → India.** This model should not be used in India without new data and re-training.
  - **Different machines.** Many Indian machines are semi-automatic or top-loading **without an
    internal heater** and wash in cold water. The 2 kW heater block the model relies on would often
    be absent, leaving only the much weaker motor pattern.
  - **Different competing loads.** Water heaters (geysers, 2–3 kW) look like a washing-machine
    heater. Air-conditioners and pumps cycle on and off.
  - **Different supply conditions.** Power cuts and inverter/UPS back-up interrupt the signal, and
    rooftop solar is spreading.
  - **Different meter data.** Indian smart meters typically record 15- or 30-minute intervals (next
    question).
  - **What to do instead:** a pilot with sub-metered Indian homes is needed before any claim.

### 3. With only 15- or 30-minute smart-meter data

A typical wash lasts about 70 minutes. Its distinctive features are minute-scale: a 5–20-minute
2 kW heating block, then the drum switching on and off every few minutes. Counting heater-sized
minutes and measuring that minute-to-minute on/off "activity" is what the model relies on most
(about 60 % of its decision-making). At 15-minute resolution the
heating block becomes one or two readings raised by about 1–2 kW, which look the same as an oven,
a shower or a run of kettle boils. The drum pattern averages away entirely. At 30 minutes a whole wash is just 2–3
slightly raised readings. **Detecting individual washes and their timing becomes largely
infeasible**, and confusion with other appliances gets much worse. What remains possible is
coarser: estimating **typical weekly or monthly laundry energy** for a household or segment,
statistically and with wide error bars, and using population-level time-of-day profiles. Any
programme built on 15/30-minute data should be validated at that resolution before launch.
