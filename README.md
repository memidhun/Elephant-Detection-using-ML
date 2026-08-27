# Automated Wildlife Monitoring & Deterrent System
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Platform](https://img.shields.io/badge/platform-Ubuntu%20%7C%20Raspberry%20Pi%20%7C%20ESP32-lightgrey)
![License](https://img.shields.io/badge/license-MIT-blue)


An end-to-end, edge-AI powered wildlife monitoring and eco-friendly deterrence solution designed for forest-edge and agricultural borders. The system uses edge-optimized deep learning to detect wild animals in real time and triggers a non-lethal acetylene-gas combustion deterrent alongside long-range LoRa alerts.
<div align="center">
  <img src="https://drive.google.com/file/d/1Wv2c09zAeAQ8S1cTwT-6d36nRJHcNyhI/view?usp=sharing" alt="Field Deployment" width="400"/>
</div>
---

## Technical Highlights

* **Edge AI & Computer Vision:** Customs-trained **LoRa11n** (optimized YOLOv11n) running locally via NCNN engine on Raspberry Pi 4B.


* **Long-Range Wireless Stack:** Off-grid RF alerts over 3–5 km using LoRa SX1278 modules.


* **Automated Physical Deterrent:** Chemically generated Acetylene gas ($\text{CaC}_2 + 2\text{H}_2\text{O} \rightarrow \text{C}_2\text{H}_2 + \text{Ca(OH)}_2$) ignited via high-voltage spark to produce high-decibel acoustic and visual deterrents.


* **Off-Grid Power System:** Autonomous operation backed by a 12V solar panel and a 20,000 mAh Li-ion battery managed via BMS.



---

## Architecture & System Flow
<div align="center">
  <img src="https://drive.google.com/file/d/1P9Jyxjw0tY9AX0epS_c3wkQHUVL4tTeK/view?usp=sharing" alt="Block Diagram"width="400"/>
</div>

The system operates across a dual-unit topology to separate edge inference from high-power physical actuation.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ TRANSMITTER UNIT (Detection Edge Node)                                  │
│ ┌───────────────┐     ┌───────────────┐     ┌─────────────────────────┐ │
│ │ USB Webcam    │ ──> │ Raspberry Pi  │ ──> │ LoRa11n Model (NCNN)    │ │
│ └───────────────┘     └───────────────┘     └────────────┬────────────┘ │
│                                                          │              │
│                                                  [Animal Detected?]     │
│                                                          │              │
│                                                 YES ┌────┴────┐         │
│                                                     │         │         │
│                                                     v         v         │
│                                              Peristaltic   LoRa SX1278  │
│                                              Water Pump    Transmitter  │
└──────────────────────────────────────────────────┬────────────┬─────────┘
                                                   │            │
                                  Chemical Feed    │            │ 433/868 MHz RF
                                  (Water + CaC2)   │            │ (3-5 km Range)
                                                   v            v
┌─────────────────────────────────────────────────────────────────────────┐
│ RECEIVER UNIT (Deterrent Actuator Node)                                 │
│                                              ┌────────────────────────┐ │
│                                              │ LoRa SX1278 Receiver   │ │
│                                              └───────────┬────────────┘ │
│                                                          │              │
│                                                          v              │
│ ┌───────────────┐     ┌───────────────┐     ┌─────────────────────────┐ │
│ │  High-Voltage │ <── │ Relay Module  │ <── │ ESP32 / Arduino Controller│ │
│ │ Spark Igniter │     └───────────────┘     └────────────┬────────────┘ │
│ └───────┬───────┘                                        │              │
│         │                                                v              │
│         v                                       ┌─────────────────┐     │
│  [Acoustic Blast                                │ TFT Display /   │     │
│   & Flash Light]                                │ Buzzer Alerts   │     │
│                                                 └─────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘

```

---

## Model Benchmark & Evaluation

Tested on a dataset of 15,000 annotated images across 5 classes (Elephant, Tiger, Monkey, Squirrel, Bat):

| Model | Accuracy (%) | mAP@0.5 (%) | Frame Rate (FPS on Pi 4B) |
| --- | --- | --- | --- |
| **LoRa11n (Proposed)** | **94.2%** | **94.1%** | **11.1 FPS** |
| YOLOv5-small | 91.5% | 90.8% | 7.1 FPS |
| EfficientDet-D0 | 89.0% | 88.2% | 5.2 FPS |
| SSD-MobileNet | 86.2% | 85.0% | 9.4 FPS |

---

## Hardware Assembly & Components

### Edge Processing & Sensors

* **Raspberry Pi 4B (4GB):** Primary AI inference engine.
* **ESP32 & Arduino Nano:** Real-time sensor input, UI driving (LVGL), and relay execution control.
* **LoRa SX1278 Modules:** Long-range wireless transceiver.
* **Sensors:** Water level sensors and HX711 weight sensors for automated material monitoring.

### Actuation & Combustion System

* **5V Peristaltic Pump:** Precise water dosing into the calcium carbide chamber.
* **24V Solenoid Valve:** Controls gas release into the ignition chamber.
* **High-Voltage Arc Generator:** Ignites accumulated acetylene gas mixture safely.

## 🧪 The Deterrent Mechanism

The system relies on an automated, controlled chemical reaction to produce a loud acoustic blast and flash, mimicking a gunshot without lethal projectiles. 

**Reaction Equation:**
`CaC₂ + 2H₂O → C₂H₂ (Acetylene Gas) + Ca(OH)₂`

When an animal is detected, the peristaltic pump drops water onto the solid calcium carbide. The resulting acetylene gas is contained until optimal pressure is reached, at which point the ESP32 triggers the high-voltage arc generator, combusting the gas.

---

---

## Repository Structure

```text
├── firmware/
│   ├── transmitter/          # ESP32/Raspberry Pi capture & LoRa code
│   └── receiver/             # Arduino/ESP32 relay timing & ignition logic
├── models/
│   ├── weights/              # Exported NCNN model binaries (.param / .bin)
│   └── train_lora11n.py      # Training script & hyperparameters
├── hardware/
│   ├── schematics/           # Circuit diagrams for transmitter & receiver
│   └── cad/                  # 3D printable designs for reaction chamber
└── README.md

```

```
