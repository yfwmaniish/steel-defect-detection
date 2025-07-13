# Steel Defect Detection - Hardware Components & Implementation

## 🔧 Hardware Components List

### Core Processing Unit
| Component | Description | Price (USD) | Specifications |
|-----------|-------------|-------------|----------------|
| **Raspberry Pi 4 Model B (8GB)** | Main processing unit | $75-85 | ARM Cortex-A72, 8GB RAM, GPIO pins |
| **NVIDIA Jetson Nano** | AI/ML processing (alternative) | $99-149 | GPU acceleration, better for AI inference |
| **Intel NUC Mini PC** | High-performance option | $300-500 | x86 architecture, more powerful |

### Camera & Imaging
| Component | Description | Price (USD) | Specifications |
|-----------|-------------|-------------|----------------|
| **Raspberry Pi Camera Module V3** | High-quality imaging | $25-35 | 12MP, autofocus, wide angle |
| **USB Industrial Camera** | Professional grade | $50-150 | Higher resolution, better optics |
| **IP Camera with PoE** | Network-enabled | $80-200 | Remote access, easy installation |

### Lighting System
| Component | Description | Price (USD) | Specifications |
|-----------|-------------|-------------|----------------|
| **LED Ring Light** | Uniform illumination | $20-40 | Adjustable brightness, cool white |
| **Industrial LED Strip** | Linear lighting | $30-60 | IP65 rated, high CRI |
| **Diffusion Panel** | Even light distribution | $15-25 | Reduces shadows and reflections |

### Input/Output & Control
| Component | Description | Price (USD) | Specifications |
|-----------|-------------|-------------|----------------|
| **7" Touchscreen Display** | User interface | $60-80 | 1024x600, capacitive touch |
| **Arduino Uno R3** | Sensor interface | $20-25 | GPIO control, sensor integration |
| **Relay Module (4-channel)** | Control external devices | $10-15 | 5V/10A switching capability |
| **Emergency Stop Button** | Safety control | $15-25 | Industrial grade, fail-safe |

### Connectivity & Storage
| Component | Description | Price (USD) | Specifications |
|-----------|-------------|-------------|----------------|
| **Wi-Fi USB Adapter** | Wireless connectivity | $15-25 | 802.11ac, dual-band |
| **Ethernet Cable (Cat6)** | Wired network | $10-20 | Reliable connection |
| **microSD Card (64GB)** | Primary storage | $15-25 | Class 10, high endurance |
| **USB 3.0 SSD (256GB)** | Fast storage | $40-60 | Better performance for AI models |

### Power & Protection
| Component | Description | Price (USD) | Specifications |
|-----------|-------------|-------------|----------------|
| **12V Power Supply (5A)** | System power | $25-35 | Regulated, industrial grade |
| **UPS Battery Backup** | Power protection | $50-100 | Uninterrupted operation |
| **Industrial Enclosure** | Environmental protection | $80-150 | IP65 rated, metal construction |

### Sensors & Accessories
| Component | Description | Price (USD) | Specifications |
|-----------|-------------|-------------|----------------|
| **Proximity Sensor** | Object detection | $10-20 | Detect steel presence |
| **Temperature Sensor** | Environment monitoring | $5-10 | Operating conditions |
| **Status LED Indicators** | Visual feedback | $10-20 | Red/Green/Yellow indicators |
| **Buzzer/Alarm** | Audio alerts | $5-15 | Defect detection alerts |

## 💰 Total Cost Estimation

### Budget Configuration (~$400-500)
- Raspberry Pi 4 + Camera + Basic Components
- Simple LED lighting
- Basic enclosure and power supply

### Professional Configuration (~$800-1200)
- NVIDIA Jetson Nano or Intel NUC
- Industrial camera with proper lighting
- Robust enclosure with UPS
- Full sensor integration

### Premium Configuration (~$1500-2500)
- High-end processing unit
- Professional imaging system
- Complete automation features
- Redundant systems and monitoring

## 🔄 System Architecture Flowchart

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Steel Sample  │───▶│   Camera +      │───▶│  Edge Computer  │
│    Positioning  │    │   Lighting      │    │ (Pi/Jetson/NUC) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Alert System  │◀───│  AI Processing  │◀───│ Image Capture & │
│ (LED/Buzzer/UI) │    │ (YOLO Model)    │    │  Preprocessing  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Logging  │    │   Result        │    │   Quality       │
│   & Reporting   │    │   Database      │    │   Control       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```
