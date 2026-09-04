# aiinfranetworking.md
ai infra networking

￼
￼
￼

￼
￼
￼
TRANSFORMER
     │
     ├── Linear Layers ────────► Tensor Cores
     │
     ├── Attention ────────────► Tensor Cores + SRAM
     │
     ├── Softmax ──────────────► Vector Units
     │
     ├── KV Cache ─────────────► HBM
     │
     ├── Multi-GPU ────────────► NVLink / NVSwitch
     │
     └── Multi-Node ───────────► InfiniBand + NCCL

┌─────────────────────────────────────┐
│         TRANSFORMER MODEL           │
│                                     │
│ Attention + MLP + LayerNorm         │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│         FRAMEWORK                   │
│ PyTorch / JAX / TensorFlow          │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│     COMPILER / OPTIMIZATION         │
│ CUDA / XLA / Triton                 │
│ Kernel Fusion / Graph Optimization  │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│       HARDWARE PRIMITIVES           │
│                                     │
│ GEMM │ Softmax │ Reduction          │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│          AI HARDWARE                │
│                                     │
│ Tensor Cores / Matrix Engines       │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│          MEMORY                     │
│ Registers → SRAM → Cache → HBM      │
└─────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│       MULTI-GPU NETWORK              │
│ NVLink → NVSwitch → RDMA            │
│ InfiniBand / RoCE                   │
└─────────────────────────────────────┘

￼


￼
￼

￼
￼
￼
￼

￼
￼
￼

￼

￼
￼
￼
￼
￼
￼
￼
￼
￼

￼
* Goodput measures the effective progress a system makes on an ML training task.
* It offers better insights into performance than raw data throughput, because it specifically tracks how much time the GPUs spend computing versus waiting for data.
Recent GPU VM families (including A3-ultra, A4-high, and A4X) leverage a hierarchical, rail-aligned network architecture to provide predictable, high-performance connectivity. This design minimizes communication overhead, allowing GPUs to focus on computation, thus improving goodput. This rail-aligned structure consists of three main components:

Sub-blocks are the foundational unit: a group of hosts physically co-located on a single rack. They provide extremely efficient, single-hop communication between any two GPUs within the rack, facilitated by a top-of-rack (ToR) switch. Hosts within a sub-block use Remote Direct Memory Access (RDMA) over Converged Ethernet (RoCE) for direct communication.
Blocks are multiple interconnected sub-blocks. Any GPU within a block is reachable in a maximum of two network hops via a non-blocking fabric, providing a high-bandwidth interconnect.
Clusters are multiple interconnected blocks, scaling to thousands of GPUs for large-scale training. Communication between blocks adds only one additional hop, maintaining high performance and predictability at massive scale.
Metadata for sub-blocks, blocks, and clusters is exposed to your orchestrators (like Slurm or Google Kubernetes Engine [GKE]) to enable optimal job placement across the hierarchy.
￼
￼

￼

Standard Fat-Tree
A fat-tree organizes switches into layers (Edge, Aggregation, and Core). When a GPU on Server A wants to talk to a GPU on Server B, the data must travel up through the local switch, into the aggregation layer, and sometimes up to the core switches before traveling back down. While it ensures that any node can technically talk to any other node with decent bandwidth, the shared paths introduce latency variations (jitter), which can stall a massive training job waiting on its slowest link.
￼
￼

￼

Multi-GPU instances require different network configurations for communication with the compute instance versus GPU-to-GPU information processing.
A separate network path (sometimes also referred to as the North-South network or Frontend network) handles all communication with the compute instance. This traffic includes disk access, intra-VM communication, internet access, access to Cloud Storage, host-level management, and communication with other Google Cloud services.
To manage this traffic, the GPU machine types use the Google Virtual NIC (gVNIC) configuration over Google Titanium NICs. Titanium NICs offload network processing tasks from the CPU, freeing the CPU to focus on your workloads. This separation helps ensure that general-purpose host traffic and dedicated GPU-to-GPU traffic use different physical interfaces, preventing them from competing for the same system resources.

￼

That assumption drives ECMP-based load balancing, symmetric bandwidth provisioning, and statistical multiplexing across many flows, or in other words, the idea that if we have many independent flows, the randomness of hashing will, over time, spread traffic evenly across all available paths.This works extremely well for microservices, east-west traffic, and other bursty, short-lived flows.
Embracing RoCE for efficiency, the AI backend leverages GPU clusters for intricate calculations, accelerating AI model training. Transferring the model's gradients between GPUs is crucial to maintaining output consistency as tasks shift. When AI/ML neural networks and data sets get large, the batch process jobs must be spread across multiple parallel GPUs in a node or between GPU nodes across a fabric using Parallelism techniques. GPU coherence protocols do not obtain ownership for written data or atomics, and they must perform synchronization between all GPU nodes in a cluster during the batch job runs. The batch job will do a local Sync and then a global send to synchronize non-local GPUs, as shown in the diagram below.
The GPUs in a cluster work together, performing calculations to achieve the desired outcomes. Synchronization occurs at the end of each batch run, ensuring all GPU nodes are in sync before starting the next batch run. This synchronization traffic is sent across all of the GPUs in the cluster, ensured through the RMDA protocol using RoCEv2 with Ethernet, referred to as "Long-Lived" or "Elephant flows." Below, we see how the batch jobs are run on a single GPU; a local sync is done and then sent outbound to connect to all the other GPUs in the GPU cluster design. Those GPUs then send their synchronization via their global send, and each GPU receives all of the Global sends from other GPUs. These are then used as a weighted average with the local sync, which finally does a Local update, and the subsequent batch job proceeds. 
￼
The congestion problem
Network congestion occurs when the volume of incoming data exceeds the outgoing link's bandwidth capacity in network switches. This often happens when multiple sources, such as GPUs, simultaneously send data to the same destination. While switch buffers can manage temporary congestion, prolonged congestion leads to buffers reaching their maximum capacity. Once the buffers fill, the buffers begin dropping new incoming packets, which hampers application performance due to the need for retransmission and the complexity of the transport protocol. To combat this, lossless networks use flow control mechanisms that halt incoming traffic before buffers overflow, thus avoiding packet loss. Nevertheless, flow control can also lead to the spread of congestion within the network.
Two networking buffering techniques called Priority Flow Control (PFC) and Explicit Congestion Notification (ECN) can be used with careful tuning of buffer queue depth to provide a lossless fabric for GPU connectivity. An in-depth look into these protocols is needed to properly design, manage, and tune the fabric to provide lossless connectivity between GPU nodes. 
There are two RoCE versions: RoCEv1 and RoCEv2.
* RoCEv1 is a link layer protocol that allows communication between any two hosts in the same layer two broadcast domain.
* RoCEv2 is a network layer protocol that implements the routing of RoCEv2 packets to allow hosts in different broadcast domains to communicate. It is encapsulated based on the UDP protocol.
RoCEv2 provides three advantages:
* Operation on routed networks ubiquitous in large data centers
* IP QoS – The DiffServ code point (DSCP), or VLAN PRI
* IP congestion – The explicit congestion notification (ECN) signal
While the delivery sequence of RoCEv2 UDP packets may be unpredictable, the RoCEv2 standard mandates that packets sharing an identical UDP source port and destination address must be delivered in the original order without reordering. The UDP layer serves solely as a means to identify connections. It is characterized by a UDP/IP quintuple—destination address (da), source address (sa), protocol, destination port (dport), and source port (sport)—which facilitates network management tasks by existing systems, such as load balancing via ECMP and network monitoring tools.
The key differences between TCP and RoCE are:
* TCP is stream-based, while RoCE is message-based, and UDP
* TCP implementations are typically software-based, while RoCE is implemented in the hardware
* TCP controls an inflight window, the number of unacknowledged bytes, while RoCE controls the transmission rate
To accommodate RoCEv2 transport, a network fabric must deliver non-blocking, zero-packet-loss throughput and zero and minimal latency while mitigating congestion-related traffic loss. AI/ML optimized fabrics are equipped with advanced features that help preserve a lossless network ecosystem, thanks to their comprehensive software and hardware telemetry capabilities for Explicit Congestion Notification (ECN) and Priority Flow Control (PFC). 
RoCEv2 also outlines the structure of Congestion Notification Packets (CNP. When ECN Congestion Experienced (CE) markings are detected, RNICs (RMDA capable NICs) dispatch CNPs to the sender to signal the need for a reduced transmission rate. 
BASIC ECN TERMINOLOGY
NP-Notification Point - the end node that receives the packets from the injector and sends back notifications to the injector for indications regarding the congestion situation.
CP-Congestion Point - the switch queue in which congestion happens
RP (Injector)-Reaction Point - the end node that performs rate limitation to prevent congestion
CNP- The RoCEv2 Congestion Notification Packet - The notification message an NP sends to the RP when it receives CE-marked packets.
ECN Threshold—configured packet buffer queue in each switch. Once past the ECN Mark Threshold (in this case, WRED max), the switch will rewrite the ECT header to 0x1. In the figure below, we can see the max and min queue depth. We always want to keep the input and output queues below the max configured.
￼
Typical WRED queue depth diagram. Courtesy Cisco Systems
Can PFC and ECN be used at the same time?
In the specialized data center setting, mainly where RoCEv2 Ethernet fabrics are employed for training AI models, ECN and PFC are crucial for congestion management. ECN offers long-term congestion feedback, facilitating the adaptive regulation of data transmission rates. Conversely, PFC swiftly reacts to congestion by halting specific traffic flows, instantly averting packet loss.
ECN contributes to the network's overall efficiency by modulating traffic flow in response to congestion signals. Meanwhile, PFC acts as a safeguard to prevent the loss of crucial, time-sensitive data during temporary congestion. When ECN is paired with PFC, the Ethernet network is equipped to handle traffic more efficiently, avoiding packet loss and ensuring a high data transfer rate. This is crucial for completing AI model training tasks on time.
Data Center Quantized Congestion Notification (DCQCN) merges the capabilities of ECN (Explicit Congestion Notification) and PFC (Priority Flow Control) to enable a fully lossless Ethernet experience. By utilizing ECN's network's ability to manage congestion through reduced transmission rates, DCQCN effectively limits the need for PFC, which halts data flow during congestion events. This synergy ensures smoother traffic flow and enhances overall network efficiency.
Next steps
In the following article, we shall examine DCQCN and how it combines the features of PFC and ECN to provide a lossless ethernet fabric for AI workloads. In the following article, we shall also dive into TOR switch design for inter-connecting GPUs for lowering latency and thereby lowering Job Completion Times (JCTs)

The injecting end station (Injector) or Reaction Point (RP) must set the ECN bits in the RoCEv2 IP header. The values of the bits are as follows: according to the RFC 3168 (ECT: ECN-Capable Transport), the injecting NIC sets the ECN field in the IP header to the value of ECT (01) (Note: Setting ECT (0) or ECT (1) is interchangeable.
* The RoCEv2 packet goes from the injector (Reaction Point) to the TOR switch with ECN=0x10. (Steps 1-2)
* The packet then goes through the spine without congestion, and the ECT header(ECN=0x10) remains unchanged (Step 3).
* The destination TOR leaf has a congested ingress queue (WRED over max); instead of dropping the packet, it will modify the ECN-capable field and turn ON the CE bit inside the IP header (0x03). (Step 4)
* The packet arrives from the network and goes to the receiver (Notification Point, in this case, a NIC). In the case of congestion, the ECN in the IP header is set to 0x03. We can see in the packet capture that the DSCP field is 0x03 CE (Congestion Experienced). (Step 5)
￼
RoCEv2 packet capture. Courtesy NVIDIA

 
 
 
 
 
 
 
 
 
 
* The receiving end station filters the packets with the CE bit turned on and the traffic type (RoCE) ON, triggers the event and releases the packet to the normal processing flow. The receiving end station RNIC will aggregate Congestion Notifications for each injector (QP) to avoid load due to traffic generation.
* One Congestion Notification Packet (CNP) is returned to the injector via the fabric once every x microseconds (the timing depends significantly on the OEM switches' operation). The receiving end station sends a CNP packet back to the injector. (Step 6)
￼
Packet capture courtesy of NVIDIA

 
 
 
 
 
 
 
 
 
* The switches treat the RoCEv2 CNP as a regular IP packet. (Step 7)
￼
CNP packet format. Image courtesy of NVIDIA

* The CNP packet arrives at the injecting end station. The injecting end station RNIC applies the corresponding rate limiter to that flow, slowing it down. (Step 8)
When only a portion of data packets are tagged with congestion notification bits, the injecting end station decreases the flow's traffic rate but keeps sending packets. Should congestion persist and buffer usage exceeds the WRED's upper limit, the switch will label all packets as experiencing congestion. Consequently, the sender gets numerous congestion notification packets, prompting a significant reduction in data transmission rate per its algorithm. This action helps alleviate congestion, allowing the buffer to clear. The traffic rate can increase again until further congestion signals are detected. It's important to note that RNICs must be used to provide end-to-end ECN and CNP capability. The switch queue depths and the RNIC queues must be manually configured and monitored during training runs to properly configure the queues and give a lossless fabric.
https://www.wwt.com/article/using-pfc-and-ecn-queuing-methods-to-create-lossless-fabrics-for-aiml#x-next-steps-7

Yes. The easiest way to understand DCB in an AI data-center network is to place PFC, ETS, DCBX, ECN, RoCE, NICs, switches, and routing against the OSI model and then see where they sit in the physical/logical topology.
1. DCB mapped to the OSI model

┌─────────────────────────────────────────────────────────┐
│ OSI Layer 7  │ Application                              │
│              │ NCCL / AI application                    │
├─────────────────────────────────────────────────────────┤
│ OSI Layer 6  │ Presentation                             │
├─────────────────────────────────────────────────────────┤
│ OSI Layer 5  │ Session                                  │
├─────────────────────────────────────────────────────────┤
│ OSI Layer 4  │ Transport                                │
│              │ UDP / TCP                                │
│              │ RoCEv2 uses UDP                          │
├─────────────────────────────────────────────────────────┤
│ OSI Layer 3  │ Network                                  │
│              │ IP, BGP, ECMP                            │
├─────────────────────────────────────────────────────────┤
│ OSI Layer 2  │ Data Link                                │
│              │ Ethernet + DCB                           │
│              │ PFC / ETS / DCBX / VLAN / 802.1p         │
├─────────────────────────────────────────────────────────┤
│ OSI Layer 1  │ Physical                                 │
│              │ Fiber / DAC / optical / electrical       │
└─────────────────────────────────────────────────────────┘

The key point
DCB primarily operates around Layer 2.
More specifically:
Technology	OSI layer	Main job
Ethernet	L2	Frame delivery
PFC	L2	Priority-based flow control
ETS	L2	Bandwidth scheduling
DCBX	L2	DCB capability/configuration exchange
VLAN / 802.1Q	L2	Traffic segmentation/priority
IP	L3	Packet addressing
BGP	L3	Routing
ECMP	L3	Path selection/load distribution
UDP	L4	Transport
RoCEv2	L3/L4 + RDMA semantics	RDMA over IP/UDP
NCCL	Application	GPU collective communication
2. Where does DCB sit in the AI networking stack?
Consider a GPU:

GPU
 │
 │ PCIe / NVLink-C2C
 ▼
NIC
 │
 │ Ethernet
 ▼
Switch
 │
 │ Ethernet
 ▼
NIC
 │
 ▼
GPU

The software/network stack is approximately:

        AI Application
              │
             NCCL
              │
          RDMA verbs
              │
           RoCEv2
              │
          UDP / IP
              │
          Ethernet
              │
      ┌───────┴────────┐
      │      DCB       │
      │                │
      │ PFC  ETS DCBX  │
      └────────────────┘
              │
          Physical

So you can think:
NCCL generates the communication → RoCE transports it → IP provides Layer-3 networking → Ethernet/DCB controls Layer-2 behavior.

3. Now let's put DCB into a topology
Imagine a simple two-node AI cluster:

       SERVER 1                         SERVER 2
   ┌─────────────┐                 ┌─────────────┐
   │ GPU0 GPU1   │                 │ GPU0 GPU1   │
   │ GPU2 GPU3   │                 │ GPU2 GPU3   │
   │     │       │                 │      │      │
   │    NIC      │                 │     NIC     │
   └─────┬───────┘                 └──────┬──────┘
         │                                 │
         │           Ethernet              │
         │                                 │
         └──────────┐       ┌──────────────┘
                    │       │
                 ┌──▼───────▼──┐
                 │   SWITCH     │
                 │              │
                 │ PFC          │
                 │ ETS          │
                 │ DCBX         │
                 │ ECN          │
                 └──────────────┘

The switch performs several different functions.
Layer 1
Physical transmission:

Fiber / DAC

Layer 2

Ethernet
   │
   ├── VLAN
   ├── 802.1p
   ├── PFC
   ├── ETS
   └── DCBX

Layer 3

IP
 │
 ├── Routing
 ├── BGP
 └── ECMP

Layer 4

UDP
 │
 └── RoCEv2


4. What happens when GPU0 sends data?
Suppose:

GPU0 SERVER1
     │
     ▼
    NIC
     │
     ▼
   Switch
     │
     ▼
    NIC
     │
     ▼
GPU0 SERVER2

Let's follow the packet.
Step 1 — Application
NCCL wants to exchange GPU data.

NCCL
 ↓
"Send this tensor"

This is application-level communication.

Step 2 — RoCE
The NIC performs RDMA communication.

NCCL
 ↓
RDMA
 ↓
RoCEv2

RoCEv2 encapsulates RDMA traffic inside UDP/IP.
Conceptually:

Ethernet Frame
┌──────────────────────────────────────────────┐
│ Ethernet │ IP │ UDP │ RoCE │ RDMA payload   │
└──────────────────────────────────────────────┘
    L2       L3    L4


5. Where does DCB come into this packet?
The Ethernet frame can carry a priority.
For example:

Ethernet frame
       │
       ▼
802.1Q / DSCP classification
       │
       ▼
Priority = 3
       │
       ▼
RoCE traffic class

The switch then puts it into a corresponding queue:

Switch
┌──────────────────────────┐
│ Queue 0 → Normal         │
│ Queue 1 → Storage        │
│ Queue 2 → VM             │
│ Queue 3 → RoCE ←─────────┤
│ Queue 4 → Management     │
└──────────────────────────┘

Now DCB mechanisms become important.

6. PFC at Layer 2
Suppose the RoCE queue becomes full.

RoCE Queue

██████████████████░
                  ↑
             almost full

The switch sends a PFC PAUSE for the RoCE priority.

             PFC PAUSE
Switch ───────────────────► NIC
             Priority 3

The NIC does:

Priority 0 → SEND
Priority 1 → SEND
Priority 2 → SEND
Priority 3 → PAUSE
Priority 4 → SEND

This is a Layer-2 flow-control mechanism.

7. ETS at Layer 2
Now suppose we have a 400-Gbps link.

             400 Gbps
                 │
       ┌─────────┼─────────┐
       │         │         │
      AI       Storage   Mgmt
      70%        20%      10%

ETS determines how transmission capacity is allocated among traffic classes.
So:
PFC = flow control
ETS = bandwidth allocation
Both are primarily Layer 2 mechanisms.

8. DCBX at Layer 2
Now imagine the NIC and switch need to agree on configuration.

NIC                         SWITCH
 │                            │
 │──── DCBX capabilities ────►│
 │                            │
 │◄── configuration ──────────│
 │                            │

For example:

RoCE priority = 3
PFC enabled = priority 3
ETS = 70%

DCBX helps exchange this DCB-related information.

9. Where does ECN fit?
This is where people frequently confuse DCB and ECN.
ECN is associated with IP/L3 congestion signaling, although congestion marking happens in the forwarding device based on queue state.
Conceptually:

                 Switch
                   │
             Queue filling
                   │
                   ▼
                  ECN
                   │
                   ▼
             Mark packet
                   │
                   ▼
               Receiver
                   │
                   ▼
          Congestion feedback
                   │
                   ▼
                Sender

Therefore:

DCB
 ├── PFC → L2
 ├── ETS → L2
 └── DCBX → L2

ECN
 └── Congestion signaling associated with IP traffic

For RoCE networks, ECN + DCQCN is typically used together with PFC.

10. Now add routing
This is critical for understanding an AI data-center topology.
Suppose we have:

             Spine 1
            /       \
Server ─ Leaf 1     Leaf 2 ─ Server
            \       /
             Spine 2

This is a leaf-spine topology.
There are two separate concerns.
Routing

BGP
 │
 └── Determines paths

ECMP
 │
 └── Distributes flows across paths

This operates primarily at Layer 3.
DCB

PFC
ETS
DCBX

Controls traffic behavior on the Ethernet links, primarily at Layer 2.

11. Complete AI leaf-spine topology
Now let's combine everything:

                         SPINE
                  ┌─────────────────┐
                  │                 │
              ┌───┴───┐         ┌───┴───┐
              │Spine 1│         │Spine 2│
              └───┬───┘         └───┬───┘
                  │ \               / │
                  │  \             /  │
              ┌───┴───┐         ┌───┴───┐
              │ Leaf 1│         │ Leaf 2│
              └───┬───┘         └───┬───┘
                  │                   │
             ┌────┴────┐        ┌────┴────┐
             │         │        │         │
          Server     Server   Server    Server
            │           │       │          │
           GPU         GPU     GPU        GPU

Layer 3

BGP
 ↓
ECMP
 ↓
Choose/distribute paths

Layer 2

Ethernet
 ↓
QoS classification
 ↓
PFC / ETS
 ↓
Queue management

Layer 4

UDP
 ↓
RoCEv2

Application

NCCL
 ↓
AllReduce / AllGather / ReduceScatter


12. The most important distinction: DCB vs topology
DCB is NOT a topology.
You can have:

Topology:
Leaf-Spine

and independently have:

DCB:
PFC + ETS + DCBX

So:

                    Network
                       │
             ┌─────────┴─────────┐
             │                   │
          Topology             Protocols
             │                   │
        Leaf-Spine        ┌──────┼───────┐
                          │      │       │
                         BGP   RoCE      DCB
                                │         │
                               UDP     PFC/ETS
                                │
                               IP


13. AI cluster: full OSI view
This is the mental model I recommend remembering:

┌───────────────────────────────────────────────┐
│ L7 │ NCCL / AI application                    │
├───────────────────────────────────────────────┤
│ L6 │ Serialization / representation            │
├───────────────────────────────────────────────┤
│ L5 │ Session                                  │
├───────────────────────────────────────────────┤
│ L4 │ UDP                                      │
│    │ RoCEv2 transport                         │
├───────────────────────────────────────────────┤
│ L3 │ IP                                       │
│    │ BGP / ECMP / routing                    │
│    │ ECN signaling                            │
├───────────────────────────────────────────────┤
│ L2 │ Ethernet                                 │
│    │ VLAN / 802.1p                            │
│    │ PFC                                      │
│    │ ETS                                      │
│    │ DCBX                                     │
├───────────────────────────────────────────────┤
│ L1 │ Fiber / DAC / optics / PHY              │
└───────────────────────────────────────────────┘


14. How everything works together
For an AI AllReduce:

              NCCL
                │
                ▼
             RoCEv2
                │
                ▼
               NIC
                │
                ▼
        ┌───────────────┐
        │ Leaf Switch   │
        │               │
        │ L2: PFC       │
        │ L2: ETS       │
        │ L2: DCBX      │
        │               │
        │ L3: ECN       │
        │ L3: BGP       │
        │ L3: ECMP      │
        └───────┬───────┘
                │
              Spine
                │
        ┌───────┴───────┐
        │               │
      Leaf             Leaf
        │               │
       NIC             NIC
        │               │
       GPU             GPU

Each technology has a different responsibility:
Component	Question it answers
NCCL	What GPU communication operation should happen?
RoCEv2	How do we perform RDMA over Ethernet/IP?
UDP/IP	How do we transport/address the packet?
BGP	Which routes exist?
ECMP	Which available path should a flow use?
ECN	Is congestion developing?
DCQCN	How should the sender reduce its rate?
PFC	How do we prevent packet loss when a priority queue is filling?
ETS	How do traffic classes share bandwidth?
DCBX	How do devices exchange DCB configuration?
Ethernet	How are frames transmitted over the link?
The one-line mental model
Topology determines WHERE the packet goes; routing determines WHICH PATH it takes; RoCE determines HOW RDMA traffic is transported; DCB determines HOW that Ethernet traffic is treated; ECN/DCQCN manages congestion; and PFC provides lossless protection.
That's the relationship you should keep in mind when moving from Ethernet → DCB → RoCE → RDMA → NCCL → multi-node GPU clusters.











NGC containers help with faster and more portable development and deployment of AI applications on GPUs.

Why?

NVIDIA NGC containers are pre-built, GPU-optimized containers that package AI frameworks, libraries, dependencies, and NVIDIA GPU acceleration together.

They provide:

Faster development — preconfigured environments.
Portability — run consistently across workstations, data centers, and clouds.
GPU optimization — CUDA, cuDNN, NCCL, TensorRT, and other NVIDIA components are integrated.
Reproducibility — fewer dependency and environment issues.
Why the others are incorrect
Option	What it actually describes
❌ Accelerated applications/tools for healthcare	A specific workload/domain, not the primary benefit of NGC containers
❌ Job scheduler for cluster resources	Cluster management/scheduling platforms, not NGC containers
❌ Cloud-native, multi-GPU platform for virtual collaboration/photorealistic simulation	NVIDIA Omniverse

Exam answer: Option 1.

Which of the following statements best addresses the considerations of running AI workloads on Ethernet networks?

AI workloads on Ethernet networks are unaffected by network latency, as most of the processing happens on the local hardware and not over the network.

AI workloads are incompatible with Ethernet networks due to their limited data processing capabilities.

AI workloads on Ethernet networks require careful consideration of network bandwidth, latency, and congestion.

Ethernet networks, which are built for general clouds or traditional data centers, provide reliable support for large workloads such as generative AI.

The correct answer is:

✅ AI workloads on Ethernet networks require careful consideration of network bandwidth, latency, and congestion.

Why?

AI workloads—especially distributed training and inference—generate huge amounts of data between GPUs and nodes. Network performance can therefore become a bottleneck.

Key considerations:

Bandwidth → GPUs need high throughput for exchanging large tensors.
Latency → Frequent GPU-to-GPU communication makes latency important.
Congestion → Multiple AI workloads can compete for network resources.
Packet loss/retransmission → Can significantly affect distributed workload performance.
Exam answer

👉 Option 3: AI workloads on Ethernet networks require careful consideration of network bandwidth, latency, and congestion.

Study Notes: Lossless Ethernet, DCB & RoCEv2 for AI Networks

These notes consolidate the concepts from your recent questions:

Lossless Network → DCB → PFC → ECN/DCQCN → RoCEv2 → UDP → RDMA Reliability → RNIC → GPU Direct RDMA → NCCL

1. Big Picture

Modern multi-node AI training requires GPUs to exchange enormous amounts of data.

                 GPU CLUSTER
                     │
                  NCCL
                     │
              Collective Ops
              AllReduce etc.
                     │
                     ▼
                   RDMA
                     │
                  RoCEv2
                     │
                UDP / IP
                     │
              Ethernet Fabric
                     │
        ┌────────────┴────────────┐
        │                         │
       PFC                   ECN/DCQCN
   Prevent drops             Control congestion
        │                         │
        └────────────┬────────────┘
                     │
                 BGP / ECMP
                     │
                  Routing
                     │
              Leaf-Spine Fabric
The fundamental idea

RoCE provides high-performance RDMA communication; DCB/QoS and congestion-control mechanisms engineer Ethernet to carry that traffic efficiently with very low packet loss.

2. What is a Lossless Network?

A lossless network is an engineered network where important traffic—particularly RoCE traffic—is protected from congestion-induced packet drops.

It does not mean:

"Packets can never be lost."

Instead:

The network is designed to prevent buffer overflow and minimize packet loss for the traffic class that requires it.

Why?

Congestion
    ↓
Switch queue fills
    ↓
Buffer overflow
    ↓
Packet drop
    ↓
RoCE retransmission
    ↓
Extra traffic
    ↓
Performance degradation

For GPU training, repeated packet loss can significantly affect collective communication.

3. Why AI Networks Need This

Consider 8 GPUs distributed across multiple servers:

GPU0 ─┐
GPU1 ─┤
GPU2 ─┤
GPU3 ─┼── Network Fabric
GPU4 ─┤
GPU5 ─┤
GPU6 ─┤
GPU7 ─┘

NCCL may perform:

AllReduce
AllGather
ReduceScatter
Broadcast

These operations can create synchronized traffic bursts.

For example:

GPU0 ───────┐
GPU1 ───────┤
GPU2 ───────┼──► Leaf ─► Spine ─► Leaf
GPU3 ───────┤
GPU4 ───────┘

Many flows may simultaneously target the same output port.

This creates incast/congestion.

4. What is DCB?

DCB = Data Center Bridging

It is a collection of Ethernet technologies designed to provide better traffic management for data-center workloads.

The important pieces are:

Technology	Purpose
PFC	Priority-based flow control
ETS	Bandwidth allocation/scheduling
DCBX	Configuration/capability exchange

A simple memory trick:

PFC = STOP
ETS = SHARE
DCBX = EXCHANGE

5. DCB and OSI Model

DCB is primarily associated with Layer 2.

Layer 7   Application       NCCL
Layer 6   Presentation
Layer 5   Session
Layer 4   Transport        UDP
Layer 3   Network          IP / BGP / ECMP / ECN
Layer 2   Data Link        Ethernet / PFC / ETS / DCBX
Layer 1   Physical         Fiber / DAC / optics / PHY

RoCEv2 spans several layers:

NCCL
 ↓
RDMA
 ↓
UDP
 ↓
IP
 ↓
Ethernet
 ↓
Physical

DCB primarily controls how Ethernet handles traffic, not where packets are routed.

6. PFC — Priority Flow Control

PFC = Priority-based Flow Control

IEEE 802.1Qbb.

Normal Ethernet flow control can pause the entire link.

PFC allows individual priorities/traffic classes to be paused.

Without PFC
Port
 │
 ├── RoCE
 ├── Storage
 ├── Management
 └── Other traffic

Congestion
    ↓
PAUSE
    ↓
Potentially entire traffic class/link behavior
With PFC
Port
 │
 ├── Priority 3 → RoCE → PAUSE
 ├── Priority 1 → Storage → Continue
 ├── Priority 0 → Normal → Continue
 └── Priority 5 → Management → Continue

So:

PFC protects a specific traffic class from packet drops by temporarily stopping upstream transmission for that priority.

7. How PFC Prevents Packet Loss

Suppose a switch queue is filling:

Incoming traffic
      │
      ▼
┌──────────────┐
│ Switch Queue │
│ ███████████  │  ← almost full
└──────────────┘

Before the queue overflows:

Switch
   │
   │ PFC PAUSE
   ▼
Upstream switch/NIC
   │
   X
 transmission temporarily stopped

This gives the congested switch time to drain its queue.

Queue:

██████████████ → PAUSE
███████████
███████
████
██

Eventually transmission resumes.

8. PFC is NOT End-to-End Congestion Control

This distinction is extremely important.

PFC is:

Hop-by-hop

Example:

NIC A
 │
 ▼
Leaf 1
 │
 ▼
Spine
 │
 ▼
Leaf 2
 │
 ▼
NIC B

If Leaf 2 becomes congested:

Leaf 2
   │
   │ PFC
   ▼
Spine

The pause propagates backward one hop at a time.

It is not an application-level end-to-end rate-control mechanism.

9. PFC Problem — Pause Storm

PFC itself can cause problems.

Imagine:

GPU A ─┐
GPU B ─┼──► Switch 1 ─► Switch 2
GPU C ─┘                  │
                          ▼
                       Congestion

If Switch 2 pauses Switch 1:

Switch 2
   ↑
 PFC PAUSE
   │
Switch 1
   ↑
 PFC PAUSE
   │
GPU/NICs

The pause can propagate backward.

This can create a PFC pause storm and head-of-line blocking.

Therefore:

Modern RoCE networks generally use PFC as a safety mechanism, not as the primary congestion-control mechanism.

10. ECN — Explicit Congestion Notification

ECN provides a more proactive mechanism.

Instead of dropping a packet:

Congestion
    ↓
Mark packet with ECN
    ↓
Receiver
    ↓
Congestion notification
    ↓
Sender slows down

Conceptually:

Sender
  │
  │ packets
  ▼
Switch
  │
  │ Queue becoming congested
  │
  │ ECN MARK
  ▼
Receiver
  │
  │ Congestion notification
  ▼
Sender
  │
  ↓
Reduce sending rate

The key idea:

ECN says "slow down" rather than waiting until the queue overflows and dropping packets.

11. DCQCN

DCQCN = Data Center Quantized Congestion Notification

It is a congestion-control mechanism designed for RoCE networks.

Simplified:

RoCE Sender
     │
     │ High-rate traffic
     ▼
Switch
     │
     │ Queue threshold exceeded
     ▼
ECN marking
     │
     ▼
Receiver
     │
     │ Congestion notification
     ▼
Sender RNIC
     │
     ▼
Reduce rate

Therefore:

ECN       → detects/signals congestion
DCQCN     → controls sender rate
PFC       → emergency protection against queue overflow
12. PFC vs ECN/DCQCN
Feature	PFC	ECN/DCQCN
Main purpose	Prevent drops	Control congestion
Action	Pause traffic	Reduce sending rate
Scope	Hop-by-hop	End-to-end
Mechanism	L2	ECN at IP + RDMA congestion control
Reaction	Reactive	More proactive
Primary goal	Protect queue	Avoid persistent congestion
Easy memory:
ECN/DCQCN = "SLOW DOWN"

PFC = "STOP TEMPORARILY"

RDMA retransmission = "RECOVER"
13. ETS

ETS = Enhanced Transmission Selection

IEEE 802.1Qaz.

ETS determines how bandwidth is shared among traffic classes.

Example:

100 Gbps link

RoCE       → 70%
Storage    → 20%
Management → 10%

Conceptually:

                 100G
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
     RoCE       Storage    Mgmt
      70G         20G       10G

So:

PFC protects traffic; ETS allocates bandwidth among traffic classes.

14. DCBX

DCBX = Data Center Bridging Capability Exchange

It allows devices to exchange DCB configuration/capabilities.

For example:

NIC ───────── Switch

NIC:
"I support PFC priority 3."

Switch:
"I support priority 3."

NIC:
"RoCE uses priority 3."

This helps ensure consistent configuration across the fabric.

15. RoCEv2

RoCE = RDMA over Converged Ethernet

RoCEv2 carries RDMA over:

Ethernet
   ↓
IP
   ↓
UDP
   ↓
RDMA

Typical packet:

┌───────────────────────┐
│ Ethernet              │
├───────────────────────┤
│ IP                    │
├───────────────────────┤
│ UDP                   │
│ Destination Port 4791 │
├───────────────────────┤
│ RDMA transport        │
├───────────────────────┤
│ RDMA payload          │
└───────────────────────┘
16. Why Does RoCEv2 Use UDP?

This is a critical concept.

You might initially think:

"UDP is unreliable, so why use it for RDMA?"

Because UDP is only the IP-routable envelope.

             RoCEv2
                │
       ┌────────┴────────┐
       │                 │
      UDP          RDMA Transport
       │                 │
  IP routing        Reliability
       │             PSN/ACK/NAK
       │            Retransmission
       ▼
   Ethernet

UDP provides:

IP routability
simple encapsulation
low protocol overhead
compatibility with IP routing infrastructure

It does not provide RDMA reliability.

17. How Can RoCEv2 Be Reliable Over UDP?

This is one of the most important points.

UDP:
No:
- retransmission
- ACK
- ordered reliable delivery
RDMA transport:
Yes:
- Packet Sequence Number (PSN)
- ACK/NAK
- retransmission
- reliable delivery for reliable RDMA transport

Therefore:

RoCEv2 uses UDP for encapsulation, while the RDMA transport provides the reliability semantics.

18. PSN — Packet Sequence Number

Suppose the sender transmits:

PSN 100
PSN 101
PSN 102
PSN 103
PSN 104

If:

100 ✓
101 ✓
102 ✗
103 ✓
104 ✓

the RDMA receiver can detect the sequence problem.

The RNIC's RDMA transport can initiate the appropriate recovery/retransmission behavior.

Conceptually:

Sender RNIC                Receiver RNIC

PSN 100 ──────────────────► ✓
PSN 101 ──────────────────► ✓
PSN 102 ───────────X
PSN 103 ──────────────────►
PSN 104 ──────────────────►
                              │
                              │ detect loss
                              ▼
                            NAK
                              │
                              ▼
Sender retransmits

The exact behavior depends on the RDMA transport state/mode.

19. Why Use RDMA Transport Instead of TCP?

TCP:

Reliable byte stream

RDMA:

Reliable memory operations

This is a major difference.

TCP
Application
     ↓
TCP
     ↓
Byte stream
     ↓
Remote application
RDMA
Application
     ↓
RDMA WRITE
     ↓
Remote memory

RDMA is designed for:

kernel bypass
direct DMA
zero-copy data movement
one-sided operations
hardware transport processing
low latency
high throughput
20. TCP Hardware NIC vs RNIC

A TCP NIC can offload:

TCP checksum
Segmentation
ACK processing
Some retransmission
Packet processing

But RDMA provides a different communication model.

TCP
send(bytes)
      ↓
Reliable byte stream
RDMA
RDMA WRITE
      │
      ├── local address
      ├── remote address
      ├── length
      └── rkey

This makes RDMA particularly suitable for GPU clusters.

21. How Does RNIC Know the Remote Address?

This is another critical concept.

Suppose GPU B registers a memory region:

Base address = 0x80000000
Length       = 1 MB
Permission   = WRITE
rkey         = ABC123

The remote application obtains the required RDMA metadata through a control/setup mechanism.

The sender can then issue:

RDMA WRITE

Remote address = 0x80001000
Length         = 64 KB
rkey           = ABC123

The remote RNIC validates this information.

22. How Does RNIC Prevent Over-Copying?

Suppose the registered region is:

Base   = 0x80000000
Length = 1 MB

Valid range:

0x80000000 → 0x800FFFFF

If the sender requests:

Address = 0x800F0000
Length  = 64 KB

the operation fits.

But:

Address = 0x800F0000
Length  = 128 KB

would cross the registered boundary.

The RNIC rejects the operation.

Conceptually:

              RNIC
                │
       ┌────────┴─────────┐
       │ Memory protection│
       └────────┬─────────┘
                │
      ┌─────────┼──────────┐
      ▼         ▼          ▼
     rkey    address     length
      │         │          │
      └─────────┼──────────┘
                ▼
          VALIDATE
                │
          ┌─────┴─────┐
          │           │
        Valid       Invalid
          │           │
          ▼           ▼
         DMA        Reject
23. rkey — Remote Key

Think of rkey as a capability/token associated with a registered memory region.

Example:

Memory Region
┌──────────────────────────────┐
│ Base: 0x80000000             │
│ Length: 1 MB                 │
│ Permission: RDMA_WRITE       │
│ rkey: ABC123                 │
└──────────────────────────────┘

The remote RNIC checks:

rkey
+
remote address
+
length
+
permissions

before allowing the operation.

Therefore, a remote sender cannot simply say:

"Write 1 GB to address 0x00000000"

and expect the RNIC to perform it.

24. Memory Safety vs Network Reliability

Keep these two concepts completely separate.

                   RDMA
                     │
          ┌──────────┴──────────┐
          │                     │
     Memory Safety          Reliability
          │                     │
       rkey                  PSN
       address               ACK/NAK
       length                retransmission
       permissions
          │                     │
          ▼                     ▼
 "Where can I access?"     "Did it arrive?"
Memory protection

Prevents:

Invalid/out-of-range memory access

RDMA reliability

Handles:

Packet loss/corruption and transport recovery

25. Complete AI Network Stack

This is the architecture I recommend memorizing:

┌───────────────────────────────────────────┐
│              AI Application               │
├───────────────────────────────────────────┤
│                  NCCL                     │
│      AllReduce / AllGather / etc.         │
├───────────────────────────────────────────┤
│              RDMA Transport               │
│       PSN / ACK / NAK / Reliability       │
├───────────────────────────────────────────┤
│                  UDP                      │
│              Port 4791                    │
├───────────────────────────────────────────┤
│                   IP                      │
│          ECN / Routing / ECMP             │
├───────────────────────────────────────────┤
│                Ethernet                   │
│       PFC / ETS / DCB / QoS               │
├───────────────────────────────────────────┤
│             Physical Network              │
│       NIC / Optics / DAC / Switches       │
└───────────────────────────────────────────┘

And alongside the data path:

                    Routing
                      │
                 BGP / ECMP
                      │
                      ▼
             ┌─────────────────┐
             │ Leaf-Spine Fabric│
             └─────────────────┘
                      │
       ┌──────────────┴──────────────┐
       │                             │
      ECN                           PFC
       │                             │
   Congestion                    Loss protection
       │                             │
       ▼                             ▼
    DCQCN                      Queue protection
26. Topology

A common AI Ethernet fabric is a leaf-spine topology:

                 SPINE
          ┌───────┼───────┐
          │       │       │
        Spine1  Spine2  Spine3
          │       │       │
       ┌──┴───────┴───────┴──┐
       │                      │
     Leaf1                  Leaf2
    ┌──┼──┐                 ┌──┼──┐
    │  │  │                 │  │  │
   GPU GPU GPU             GPU GPU GPU
   Node Node Node           Node Node Node
Important distinction
Topology
   ↓
Physical/logical connectivity

Routing
   ↓
Which path should packet take?

RoCE
   ↓
How is RDMA transported?

DCB
   ↓
How is Ethernet traffic treated?

ECN/DCQCN
   ↓
How is congestion controlled?

PFC
   ↓
How are queue overflows prevented?

RNIC
   ↓
How are RDMA operations executed?
27. End-to-End Packet Journey

Imagine GPU A sends data to GPU B.

GPU A
 │
 │ NCCL
 ▼
RNIC A
 │
 │ RDMA operation
 │ PSN
 ▼
RoCEv2
 │
 │ UDP
 ▼
IP
 │
 │ BGP/ECMP-selected path
 ▼
Leaf
 │
 │ Queue
 │
 ├── ECN → mark if congestion
 │
 └── PFC → pause if queue danger
 │
 ▼
Spine
 │
 ▼
Leaf
 │
 ▼
RNIC B
 │
 │ validate rkey/address/length
 ▼
GPU B memory
28. Complete Congestion Scenario

Suppose thousands of GPU flows converge on one switch port.

GPU0 ─┐
GPU1 ─┤
GPU2 ─┤
GPU3 ─┼──► Switch ───► Output
GPU4 ─┤                 │
GPU5 ─┘                 ▼
                    Queue fills
Stage 1 — ECN
Queue increasing
      ↓
ECN threshold reached
      ↓
Packets marked
Stage 2 — DCQCN
Receiver observes congestion
      ↓
Congestion notification
      ↓
Sender RNIC reduces rate
Stage 3 — PFC

If congestion becomes severe:

Queue approaching overflow
      ↓
PFC PAUSE
      ↓
Upstream transmission stops temporarily
Stage 4 — Recovery
Queue drains
      ↓
PFC resumes traffic
      ↓
DCQCN gradually increases rate

This is the intended hierarchy:

             Congestion
                  │
                  ▼
             ECN marking
                  │
                  ▼
              DCQCN
            slow sender
                  │
          still congested?
                  │
                  ▼
                 PFC
          protect queue
                  │
             packet loss?
                  │
                  ▼
        RDMA retransmission
29. Most Important Distinctions
DCB vs RoCE
DCB = Ethernet traffic management
RoCE = RDMA over Ethernet
PFC vs ECN
PFC = pause
ECN = mark
ECN vs DCQCN
ECN = congestion signal
DCQCN = sender rate control
PFC vs retransmission
PFC = prevent loss
Retransmission = recover from loss
UDP vs RDMA reliability
UDP = routing/encapsulation
RDMA = transport reliability
TCP vs RDMA
TCP = reliable byte stream
RDMA = direct memory operations
rkey vs PSN
rkey = memory protection
PSN  = packet sequencing/reliability
30. Final Mental Model

If you remember only this diagram, remember this:

                     AI WORKLOAD
                          │
                        NCCL
                          │
                   GPU COLLECTIVES
                          │
                          ▼
                    ┌──────────┐
                    │   RDMA   │
                    │          │
                    │ PSN      │
                    │ ACK/NAK  │
                    │ Retrans. │
                    └────┬─────┘
                         │
                      RoCEv2
                         │
                      UDP/IP
                         │
             ┌───────────┴───────────┐
             │                       │
          Routing                Congestion
          BGP/ECMP                    │
             │                 ┌──────┴──────┐
             │                 │             │
             │                ECN           PFC
             │                 │             │
             │              DCQCN         Pause
             │                 │             │
             └────────────┬────┴─────────────┘
                          │
                     Ethernet/DCB
                          │
                    Leaf-Spine Fabric
                          │
                         RNIC
                          │
                    GPU Direct RDMA
                          │
                      GPU Memory
⭐ One-line summary

NCCL generates GPU communication → RDMA provides direct memory-based transport and reliability → RoCEv2 puts RDMA inside UDP/IP → BGP/ECMP selects paths → ECN/DCQCN controls congestion → PFC protects queues from overflow → RNIC performs the hardware DMA into the destination GPU memory.

This gives you the conceptual chain:

GPU → NCCL → RDMA → RoCEv2 → UDP/IP → Ethernet/DCB → ECN/DCQCN + PFC → BGP/ECMP → RNIC → GPU memory.

CONTEXT : STAFF ENGINEER INTERVIEW SHARE MORE QUESTIONS
👍

Absolutely. For a Staff Engineer interview, you should expect questions that go beyond definitions. Interviewers will test whether you can design, reason about trade-offs, debug failures, and explain the architecture end-to-end.

Based on your current focus on GPU networking → NCCL → RDMA → RoCEv2 → DCB → lossless Ethernet → multi-node AI clusters, I would prepare these questions.

Staff Engineer Interview — AI Networking Question Bank
1. Lossless Ethernet / RoCE — Core
What does lossless Ethernet actually mean?
Why does RoCE need a lossless or low-loss network?
Is RoCE inherently lossless?
Why is packet loss particularly expensive for distributed AI training?
What happens when a RoCE packet is dropped?
If RoCE has retransmission, why do we need PFC?

"RoCEv2 provides reliability through end-to-end mechanisms such as ACK/NACK and retransmission, but retransmission is reactive. PFC is a link-level mechanism used to prevent buffer overflow and packet loss for the loss-sensitive RoCE traffic class. In a well-designed RoCE network, ECN/DCQCN provides congestion control, while PFC provides a loss-avoidance safety mechanism; retransmission handles residual losses."

 RoCE Network
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
         ECN        PFC     Retransmission
          │          │          │
          ▼          ▼          ▼
      Congestion   Prevent     Recover
      signaling    drops       lost packets
          │
          ▼
       DCQCN
          │
          ▼
    Reduce sender rate

    The answer is: ECN/DCQCN is a feedback-control mechanism; PFC is a last-resort loss-prevention mechanism. They operate at different timescales and have different failure modes.

1. ECN/DCQCN reacts, but not instantaneously

Consider:

GPU Sender
   │
   │ packets at 400 Gb/s
   ▼
Switch
   │
   │ Queue building
   ▼
Receiver

The sequence is roughly:

Queue starts growing
       ↓
Switch detects threshold
       ↓
ECN marks packets
       ↓
Receiver sees ECN
       ↓
Congestion notification
       ↓
Sender receives feedback
       ↓
DCQCN reduces rate

There is a feedback delay.

During that delay, packets continue arriving.

If the available buffer/headroom is insufficient:

Queue
████████████████████████
                         ↑
                    buffer full
                         ↓
                       DROP

PFC is designed to prevent that final step.

2. PFC provides a local, fast reaction

PFC doesn't need an end-to-end feedback loop.

The congested switch can essentially say:

             PFC
              ↓
Sender ───────────────► Switch
   ▲                     │
   │                     │
   └──── PAUSE ◄─────────┘

The sender temporarily stops transmitting that priority.

So:

ECN/DCQCN:
"Please slow down because congestion is developing."

PFC:
"STOP NOW because my buffer is almost full."

That's the fundamental difference.

3. Why is packet loss particularly undesirable for RoCE?

TCP is designed around packet loss:

TCP
packet loss
   ↓
retransmission
   ↓
continue

RoCE can also recover from lost packets, but RDMA has much tighter latency/performance requirements.

For example:

GPU0 ──┐
GPU1 ──┤
GPU2 ──┼── AllReduce
GPU3 ──┘

Suppose one flow experiences loss:
PFC itself can create problems.

Imagine:

             Switch S1
                │
         PFC PAUSE
                │
                ▼
             Switch S0
          ┌───────────┐
          │   Queue   │
          └───────────┘
             │
       other traffic

The pause propagates upstream.

You can get:

Congested receiver
       ↓
     PFC
       ↓
 upstream switch pauses
       ↓
 more queues build
       ↓
 other flows affected
       ↓
 congestion spreads

This is called PFC head-of-line blocking / congestion spreading.

In extreme cases you can get a PFC storm.

Therefore, modern RoCE networks generally don't want PFC to be the primary congestion-control mechanism.

5. Then what is the ideal division of responsibility?

Think of it as three layers:

                 RoCE congestion architecture
                         │
              ┌──────────┴──────────┐
              │                     │
         ECN + DCQCN              PFC
              │                     │
       Primary mechanism       Safety mechanism
       for congestion          against packet loss
              │                     │
              └──────────┬──────────┘
                         │
                   Retransmission
                         │
                   Last resort
ECN/DCQCN

Normal operation

Queue increasing
     ↓
ECN marking
     ↓
DCQCN
     ↓
reduce injection rate
PFC

Emergency protection

Queue approaching exhaustion
     ↓
PFC
     ↓
temporarily stop traffic
     ↓
prevent packet drop
Retransmission

Failure recovery

Packet actually lost
     ↓
detect loss
     ↓
retransmit
6. The critical concept: headroom

This is probably the most important concept for your interview.

PFC requires the switch to maintain enough buffer headroom to absorb packets that are already in flight after sending PAUSE.

Suppose:

Sender ─────── 400 Gb/s ───────► Switch

The switch sends PFC PAUSE, but packets already transmitted cannot magically disappear.

Therefore:

PFC threshold
       │
       ▼
██████████████████
       │
       │  headroom
       ▼
████████
       │
       ▼
    buffer limit

You need sufficient headroom for:

propagation delay
serialization delay
switch processing
NIC response time
packets already in flight

If headroom is badly configured:

PFC triggered
      ↓
buffer still fills
      ↓
DROP

So PFC does not guarantee losslessness by itself.

7. Why not simply make ECN extremely aggressive?

You might ask:

"Why not set the ECN threshold very low and eliminate PFC?"

You can reduce the probability of overflow, but you cannot completely eliminate the problem.

There are several reasons:

Feedback is delayed
ECN → receiver → sender → rate reduction

takes time.

Traffic can be bursty

A GPU/NIC can inject traffic very rapidly.

Multiple senders can converge simultaneously
GPU0 ─┐
GPU1 ─┤
GPU2 ─┼──► Switch ──► GPU10
GPU3 ─┤
GPU4 ─┘

Even if each sender individually reacts correctly, the aggregate arrival rate can temporarily exceed the output capacity.

Control loops aren't perfect

DCQCN is a distributed feedback-control algorithm. Multiple flows reacting simultaneously can create oscillations or transient queue growth.

PFC provides the hard safety boundary.

8. Best mental model

Think of a car:

              Car
               │
      ┌────────┼────────┐
      │        │        │
      ▼        ▼        ▼
 Accelerator  Brake   Crash protection
      │        │        │
      │        │        │
      ▼        ▼        ▼
 ECN/DCQCN    PFC   Retransmission

More accurately:

ECN/DCQCN
   ↓
"Slow down"

PFC
   ↓
"Stop temporarily"

Retransmission
   ↓
"Something went wrong; recover"
Staff Engineer interview answer

I'd answer it like this:

"ECN/DCQCN is the primary congestion-control mechanism in a RoCE network, but it is an end-to-end feedback loop and therefore has reaction latency. During bursts or synchronized GPU traffic, the queue can continue growing after ECN marking and before the sender reduces its rate. PFC provides a local, fast backpressure mechanism that prevents the loss-sensitive RoCE queue from exhausting its buffer. So ECN/DCQCN handles congestion proactively, PFC provides a loss-prevention safety net, and RoCE retransmission handles residual packet loss. However, PFC must be carefully scoped and engineered with sufficient headroom because excessive PFC can cause head-of-line blocking and congestion spreading."

The interview answer I'd give

"In RoCE, the switch acts as the congestion point and marks ECN CE when the queue crosses its congestion threshold. The receiving NIC detects the CE-marked RoCE packet and generates a CNP back to the sender. The sender NIC's DCQCN reaction point maintains a congestion state, typically alpha, and uses it to reduce the per-flow injection rate. When CNPs disappear, the NIC gradually increases the rate again.

The thresholds are fundamentally a buffer-and-feedback-delay problem. ECN must be triggered early enough that the end-to-end control loop can react before the queue exhausts. A first-order estimate of required headroom is incoming bandwidth multiplied by the worst-case reaction time, with additional allowance for packets already in flight, switch/NIC processing and burstiness. PFC is then configured closer to buffer exhaustion, leaving enough headroom for traffic that continues arriving after the pause is generated.

So ECN/DCQCN is the primary closed-loop congestion-control mechanism, while PFC is the fast local safety mechanism against buffer overflow."

One formula to remember
Required headroom ≈
        Incoming bandwidth
        ×
        Worst-case reaction time

        nd the conceptual ordering:

        ECN
         ↓
   DCQCN reacts
         ↓
   queue controlled
         ↓
      normally
         
If reaction is too slow:

        PFC
         ↓
   protect headroom

If everything fails:

   packet loss
         ↓
   retransmission

One nuance: there is no single standards-mandated “correct” ECN or PFC threshold formula for every RoCE fabric. ECN marking behavior is an implementation/configuration choice, and real deployments tune thresholds to their switch buffer architecture, topology, workload burstiness and desired latency.

You shouldn't blindly use RTT.

The actual required headroom is closer to:

Headroom ≥
    packets already in flight
  + propagation delay
  + serialization delay
  + switch processing
  + PFC generation/processing
  + NIC reaction delay

A simplified engineering approximation:

H ≈ R × T_reaction

where:

H = required headroom
R = incoming rate
T = reaction time

For multiple simultaneous senders:

H ≈ Σ Ri × Ti

or, conservatively:

H ≈ Aggregate arrival rate × worst-case reaction time

2. Step 1 — Switch detects congestion

Suppose a 400-Gb/s switch port is transmitting toward a receiver, but several GPUs are sending toward it:

GPU0 ──────┐
GPU1 ──────┤
GPU2 ──────┼────► Switch ─────► GPU10
GPU3 ──────┤
GPU4 ──────┘

The output queue starts increasing:

Queue:

10 KB
20 KB
40 KB
80 KB
120 KB
...

The switch has an ECN marking policy.

A simple example:

Kmin = 100 KB
Kmax = 200 KB

Then:

Queue < 100 KB
       │
       └── no ECN

100–200 KB
       │
       └── increasing ECN marking probability

> 200 KB
       │
       └── essentially all eligible packets marked

Actual implementations can use static or dynamic thresholds and different marking algorithms. NVIDIA documents both minimum/maximum ECN thresholds and dynamic ECN mechanisms.

3. Step 2 — Switch sets the ECN CE bit

RoCEv2 runs over IP, so the switch can mark the packet's IP header with:

ECN = CE

Conceptually:

Before:

RoCE packet
┌─────────────────────────┐
│ IP │ UDP │ RoCE │ Data  │
└─────────────────────────┘
       ECN = ECT


After switch:

┌─────────────────────────┐
│ IP │ UDP │ RoCE │ Data  │
└─────────────────────────┘
       ECN = CE
             ↑
       congestion!

The important point:

The switch does not directly tell the sender to reduce its rate. It marks the packet.

4. Step 3 — Receiver sees the CE-marked packet

The destination NIC receives:

RoCE packet
      │
      ▼
ECN = CE
      │
      ▼
Notification Point

The receiver's RoCE congestion-control logic recognizes that the packet encountered congestion.

It then generates a CNP — Congestion Notification Packet back toward the sender.

Sender                                      Receiver

   ──────── data ───────────────────────────►
             ECN = CE

   ◄──────────── CNP ───────────────────────

This is an important distinction:

CNP is not the same thing as the ECN-marked data packet.

The switch marks the data packet.

The receiver generates the CNP.

5. Step 4 — CNP reaches the sender NIC

Now the sender NIC receives:

CNP
 │
 ▼
Reaction Point (RP)
 │
 ▼
DCQCN state

This happens inside the NIC hardware/firmware, rather than requiring the CPU to process every congestion notification.

Modern NVIDIA adapters expose DCQCN parameters for this behavior.

6. Step 5 — DCQCN maintains α

This is the interesting part.

DCQCN maintains a congestion state variable called:

α (alpha)

Think of α as:

α ≈ "How serious does congestion appear to be?"

If CNPs keep arriving:

CNP
 ↓
α increases
 ↓
CNP
 ↓
α increases
 ↓
CNP
 ↓
α increases

If congestion disappears:

No CNP
 ↓
α decreases
 ↓
No CNP
 ↓
α decreases

NVIDIA's implementation updates α periodically, with the update period tied to a configured RTT-related parameter.

A simplified conceptual equation is:

CNP received:

αnew = (1-g) αold + g

No CNP:

αnew = (1-g) αold

So α becomes a smoothed estimate of congestion severity.

7. Step 6 — α determines the rate reduction

Now the sender's NIC calculates a new transmission rate.

A simplified DCQCN-style relationship is:

New Rate ≈ Old Rate × (1 - α / 2^g_d)

subject to configured limits.

NVIDIA documents the rate-decrease logic using α together with parameters such as RPG_GD, RPG_MIN_DEC_FAC, and RPG_MIN_RATE.

For example, conceptually:

Current rate = 400 Gb/s
α = small
       ↓
small reduction

Current rate = 400 Gb/s
α = large
       ↓
large reduction

So:

ECN frequency/severity
        ↓
       CNP
        ↓
       α
        ↓
rate reduction

9. Now the really important part: threshold calculation

There isn't one universal formula for RoCE ECN/PFC thresholds.

They're normally engineered from:

link speed
RTT
switch pipeline latency
NIC reaction time
propagation delay
packet size
number of senders
buffer architecture
burst size
desired queueing latency

The basic physical constraint is:

How many bytes can arrive during the time it takes the congestion-control mechanism to react?
What happens if PFC is disabled?
What happens if PFC is enabled everywhere?
What is a PFC storm?
What is head-of-line blocking?
How do you prevent PFC from propagating congestion across the fabric?
How much buffer/headroom is required for PFC?
What causes PFC deadlock?
How would you troubleshoot a network experiencing excessive PFC pause frames?
How would you determine whether packet loss is caused by congestion or a physical link problem?
        Retransmissions
                          │
              ┌───────────┴───────────┐
              │                       │
         Congestion?             Physical?
              │                       │
        ECN ↑ / PFC ↑            CRC/FCS ↑
        Queue ↑                  FEC ↑
        CNP ↑                    symbol ↑
        DCQCN ↓rate             optical anomalies
        "I would first localize the PFC events by switch, port and priority, then correlate PFC with queue occupancy, ECN marking, CNPs and DCQCN rate reduction. If queue depth, ECN, CNP and PFC rise together while physical error counters remain clean, I would treat it as congestion and investigate oversubscription, traffic concentration, burstiness and PFC propagation.

For packet loss, I would separate congestion loss from physical loss by correlating RoCE retransmissions with queue/ECN/PFC counters versus FEC, CRC/FCS, PCS, symbol and optical counters. Congestion typically produces queue buildup, ECN and PFC without physical errors; a bad link produces FEC/PCS/CRC/symbol errors even when queues are healthy.

Finally, I would trace the PFC pause upstream to determine whether the original congestion is localized or whether PFC itself is propagating congestion through the fabric. I would only tune PFC/ECN thresholds after identifying the actual root cause."

The mental model to memorize
                 PACKET LOSS
                     │
             ┌───────┴────────┐
             │                │
        CONGESTION          PHYSICAL
             │                │
       Queue ↑              FEC ↑
       ECN ↑                PCS ↑
       CNP ↑                CRC ↑
       PFC ↑                Symbol ↑
       DCQCN ↓rate         Optical issue
             │                │
             ▼                ▼
       Fix traffic/       Fix cable/optic/
       congestion         port/PHY

       

3. DCB — Staff-Level Questions
What is DCB and why was it introduced?
Explain:
PFC
ETS
DCBX
Where does DCB operate in the OSI model?
Is DCB a routing protocol?
How is DCB different from QoS?
What happens if the NIC and switch have different PFC configurations?
What is the relationship between:
Priority
Traffic Class
PFC
ETS
DSCP
802.1p
How would you map RoCE traffic to a lossless priority?
What happens if RoCE traffic shares a queue with TCP traffic?
How would you design QoS for:
RoCE
Storage
Management
Control traffic
What is ETS actually solving?
Why isn't PFC sufficient by itself?
Is DCBX mandatory for a RoCE deployment?
4. ECN / DCQCN — Very Important
Explain ECN from packet generation to congestion response.
Where is ECN implemented?
What exactly does the switch do when congestion occurs?
How does the receiver know that congestion occurred?
What is a CNP?
What is DCQCN?
Explain:
ECN → CNP → DCQCN → Rate reduction
What is the difference between PFC and ECN?
Why use both PFC and ECN?
What happens if ECN thresholds are configured too high?
What happens if ECN thresholds are too low?
What happens if PFC triggers before ECN can control the sender?
How would you tune ECN/DCQCN for an AI cluster?
How would you identify whether DCQCN is actually reducing congestion?
5. RoCEv2 — Deep Questions
Explain the complete RoCEv2 packet structure.
Ethernet
   ↓
IP
   ↓
UDP
   ↓
RDMA
Why does RoCEv2 use UDP instead of TCP?
UDP is unreliable. How does RoCEv2 provide reliable delivery?
What is the role of PSN?
What happens when PSN 100 is followed by PSN 102?
Where does retransmission happen?
Is retransmission performed by the CPU?
What is the role of the RNIC?
Why is UDP port 4791 used?
What is the difference between RoCEv1 and RoCEv2?
Why is RoCEv2 routable while RoCEv1 isn't?
Can RoCEv2 cross a Layer-3 boundary?
Can RoCEv2 use ECMP?
What happens to RoCE packets when they traverse multiple routers?
What fields are used to identify a RoCE flow?
How does ECMP affect distributed AI traffic?
6. RDMA — Staff-Level Questions
What exactly does RDMA mean?
What is kernel bypass?
What is zero-copy?
What is one-sided communication?
Difference between:
RDMA READ
RDMA WRITE
SEND/RECEIVE
Why is RDMA faster than TCP?
Can TCP NIC hardware communicate directly with GPU memory?
If yes, why is RDMA still preferred?
What is the difference between:
TCP byte stream
vs
RDMA memory operation
Where is RDMA transport implemented?
What does the RNIC do?
What is a Queue Pair?
What are:
QP
CQ
WQE
WR
MR
lkey
rkey
What is memory registration?
Why does RDMA require memory registration?
How does an RNIC know the remote address?
How does the RNIC prevent an RDMA WRITE from overwriting memory beyond the buffer?
What happens if the sender provides an invalid rkey?
What happens if:
remote_address + length > registered_region
What is the difference between lkey and rkey?
How does RDMA provide memory protection?
How does RDMA interact with the IOMMU?
7. GPU Direct RDMA

These are particularly good Staff Engineer-level questions.

How does an RNIC access GPU memory?
What is GPUDirect RDMA?
Explain:
GPU
 ↓
PCIe
 ↓
RNIC
 ↓
Network
 ↓
RNIC
 ↓
PCIe
 ↓
GPU
Where does CPU involvement remain?
Does the RNIC understand CUDA?
How does GPU virtual memory get translated for DMA?
What happens if GPU memory is not registered?
Why can't a normal NIC arbitrarily DMA into GPU memory?
What role does PCIe play?
What happens if the NIC and GPU are connected through different PCIe root complexes?
How does NUMA topology affect GPU networking?
Why is GPU ↔ NIC affinity important?
How would you troubleshoot poor GPUDirect RDMA performance?
7. NCCL + Networking
What is NCCL?
Why does NCCL use RDMA?
What happens during NCCL AllReduce?
Explain:
GPU
 ↓
NCCL
 ↓
RNIC
 ↓
RoCE
 ↓
Switch
 ↓
RNIC
 ↓
GPU
What happens if one GPU is significantly slower than the others?
Why can network congestion slow down the entire training job?
What is the relationship between:
NCCL
RDMA
RoCE
RNIC
PFC
ECN
DCQCN
How would you troubleshoot poor NCCL performance?
How do you distinguish:
GPU bottleneck
PCIe bottleneck
NIC bottleneck
Network bottleneck
Why does distributed training care about tail latency, not just average latency?
Why can a single slow network link affect an entire collective operation?
8. AI Cluster Architecture
Design a 1,000-GPU Ethernet AI cluster.
Would you use:
Leaf-Spine
Fat Tree
Dragonfly
Clos

and why?

How many leaf switches would you use?
How would you calculate oversubscription?
What does 1:1 non-blocking mean?
What is a rail-optimized topology?
What is a GPU rail?
Why might you use multiple NICs per GPU server?
How do you map:
GPU → NIC → Leaf → Spine
How would you design redundancy?
What happens when a leaf switch fails?
What happens when a spine switch fails?
How does ECMP provide load balancing?
Why can ECMP create problems for elephant flows?
How would you design the network to handle synchronized AllReduce traffic?
9. Routing Questions
Why use BGP in an AI Ethernet fabric?
BGP vs OSPF — which would you choose and why?
Why is BGP commonly used in leaf-spine data centers?
What is ECMP?
How does ECMP select a path?
Why can two packets belonging to the same flow normally follow the same path?
What happens when a spine link fails?
How quickly can the network reconverge?
What is the relationship between:
BGP
ECMP
RoCE
PFC
ECN
Is PFC involved in routing?
Is ECN involved in routing?

Staff-level trap:
The correct answer is generally:

BGP/ECMP → PATH SELECTION

RoCE → TRANSPORT

PFC/ETS → L2 TRAFFIC CONTROL

ECN/DCQCN → CONGESTION CONTROL
10. Troubleshooting Scenarios

These are more important than definition questions for Staff Engineer interviews.

Scenario 1

NCCL AllReduce performance suddenly drops by 40%. GPU utilization is only 60%. How do you debug it?

You should think:

GPU
 ↓
PCIe
 ↓
NIC
 ↓
Network
 ↓
PFC
 ↓
ECN
 ↓
DCQCN
 ↓
Packet drops
 ↓
Retransmission
Scenario 2

One switch shows millions of PFC pause frames. What could be happening?

Investigate:

Queue congestion
     ↓
ECN configuration
     ↓
DCQCN behavior
     ↓
PFC threshold
     ↓
Buffer/headroom
     ↓
Oversubscription
     ↓
Hotspot
Scenario 3

There are no packet drops, but NCCL performance is poor.

This is a great Staff question.

Don't immediately conclude:

"Network is healthy."

Investigate:

PFC pause time
ECN marks
DCQCN rate reduction
Queue occupancy
Buffer utilization
Link utilization
Retransmissions
NIC counters
PCIe bandwidth
GPU-NIC affinity
NUMA
Scenario 4

RoCE traffic is experiencing high latency even though bandwidth utilization is only 50%.

Possible causes:

microbursts
queue buildup
PFC
ECN threshold
poor QoS
elephant-flow hashing
congestion hotspot
PCIe/NIC issue
retransmissions
Scenario 5

One GPU is consistently slower during AllReduce.

How do you isolate the problem?

GPU
 ↓
GPU-NIC affinity
 ↓
PCIe
 ↓
NIC
 ↓
NIC counters
 ↓
Network path
 ↓
Switch port
 ↓
PFC/ECN
 ↓
Remote NIC
 ↓
Remote GPU
11. Architecture / Design Questions

These are the questions I'd prioritize for a Staff Engineer interview.

Q1

Design a lossless Ethernet fabric for a 10,000-GPU AI cluster.

You should discuss:

Topology
   ↓
Leaf-Spine / Clos
   ↓
BGP/ECMP
   ↓
RoCEv2
   ↓
PFC
   ↓
ECN/DCQCN
   ↓
QoS/ETS
   ↓
Buffers
   ↓
NIC/GPU affinity
   ↓
Observability
Q2

Design a network that supports both AI training and normal enterprise traffic.

You need to discuss:

Traffic classes
     ↓
QoS
     ↓
PFC
     ↓
ETS
     ↓
ECN
     ↓
Isolation
Q3

Your RoCE fabric is experiencing PFC storms. Redesign it.

A strong answer should mention:

Don't rely solely on PFC
        ↓
Use ECN/DCQCN proactively
        ↓
Tune thresholds
        ↓
Limit lossless classes
        ↓
Correct buffer/headroom
        ↓
Avoid unnecessary PFC propagation
        ↓
Check topology/oversubscription
12. The "Why?" Questions Interviewers Love

Be ready for these chains:

Why RoCE?

Why not TCP?

↓

Why UDP?

Why use an unreliable transport?

↓

Why RDMA reliability?

If UDP is unreliable, where does reliability come from?

↓

Why PFC?

If RDMA can retransmit, why prevent packet loss?

↓

Why ECN?

If PFC prevents packet loss, why do we need ECN?

↓

Why DCQCN?

If ECN marks packets, who actually slows the sender?

↓

Why RNIC?

Why can't a normal NIC do this?

↓

Why GPU Direct RDMA?

Why can't the CPU simply copy the data?

↓

Why NCCL?

Why can't the application just use TCP sockets?

That entire chain is excellent Staff Engineer interview preparation.

13. The Ultimate Interview Diagram

You should be able to draw this on a whiteboard in 2–3 minutes:

                         AI APPLICATION
                               │
                              NCCL
                               │
                    AllReduce / AllGather
                               │
                               ▼
                         GPU MEMORY
                               │
                     GPU Direct RDMA
                               │
                               ▼
                             RNIC
                               │
                     ┌─────────┴─────────┐
                     │     RoCEv2        │
                     │                   │
                     │ RDMA Transport    │
                     │ PSN / ACK / NAK   │
                     │ Retransmission    │
                     └─────────┬─────────┘
                               │
                              UDP
                               │
                               IP
                    ┌──────────┴──────────┐
                    │                     │
                  ECN                  BGP/ECMP
                    │                     │
                 DCQCN               Path Selection
                    │
                    ▼
                 Ethernet
                    │
              ┌─────┴─────┐
              │    DCB    │
              │            │
              │ PFC        │
              │ ETS        │
              │ DCBX       │
              └─────┬──────┘
                    │
              Leaf-Spine Fabric
                    │
                    ▼
                   RNIC
                    │
            GPU Direct RDMA
                    │
                    ▼
                GPU MEMORY
The Staff Engineer answer in one sentence

"I would design RoCEv2 as a high-performance RDMA transport over a routed Ethernet Clos fabric, using BGP/ECMP for path selection, ECN/DCQCN for proactive end-to-end congestion control, PFC as a bounded hop-by-hop safety mechanism against queue overflow, ETS for bandwidth allocation, and GPUDirect RDMA so the RNIC can move data directly between GPU memory regions while enforcing rkey-based memory protection."

If you can explain and defend every box in that sentence, you're moving from a networking engineer answer toward a Staff Engineer architecture answer.

HOW WOULD YOU EXPLAIN THE INTEGRATION WITH SDWAN CONTROL PLANE

For a Staff Engineer interview, I would explain SD-WAN control-plane integration by separating control plane, data plane, and policy/intent plane.

1. High-level architecture
                    ┌──────────────────────────────┐
                    │        SD-WAN Controller      │
                    │                              │
                    │  • Topology / discovery      │
                    │  • Routing policy            │
                    │  • Path selection            │
                    │  • Security policy           │
                    │  • QoS / SLA policy          │
                    └──────────────┬───────────────┘
                                   │
                         Control Plane
                    routes / policy / intent
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
        ┌─────▼─────┐                             ┌─────▼─────┐
        │ SD-WAN PE │                             │ SD-WAN PE │
        │ / Edge A  │                             │ / Edge B  │
        └─────┬─────┘                             └─────┬─────┘
              │                                         │
              │             DATA PLANE                  │
              └─────────────── Overlay ────────────────┘
                       IPsec / GRE / VXLAN

The key interview statement is:

The SD-WAN controller does not normally forward user packets. It computes and distributes policy, topology, and path information; the SD-WAN edges make forwarding decisions and carry the actual traffic.

2. Where does the SD-WAN control plane fit?

Think of the system as three layers:

             ┌─────────────────────────────┐
             │       Intent / Policy       │
             │ "Voice via MPLS, Internet   │
             │  via broadband"              │
             └──────────────┬──────────────┘
                            │
                            ▼
             ┌─────────────────────────────┐
             │      SD-WAN CONTROL PLANE   │
             │                             │
             │ topology                    │
             │ routes                      │
             │ policy                      │
             │ path selection              │
             │ security/QoS                │
             └──────────────┬──────────────┘
                            │
                     programmed into
                            │
                            ▼
             ┌─────────────────────────────┐
             │       SD-WAN EDGE           │
             │                             │
             │ routing tables              │
             │ forwarding tables           │
             │ tunnels                     │
             │ QoS                         │
             │ ACL/firewall                │
             └──────────────┬──────────────┘
                            │
                            ▼
                     Actual packets
3. What does the controller actually do?

Suppose a branch has:

             Branch
                |
       ┌────────┼────────┐
       │        │        │
      MPLS   Internet   5G
       │        │        │
      10ms     30ms     50ms
      0% loss  1% loss  3% loss

The application policy says:

Critical traffic must use the best available path.

The SD-WAN controller collects:

topology
reachability
link status
latency
packet loss
jitter
bandwidth
tunnel state
security information
application classification
SLA information

It then distributes policy/path information to the edges.

The edge ultimately forwards the packet.

4. Control plane vs data plane

This distinction is extremely important in interviews.

Control plane
Edge A ──────► Controller
       topology
       routes
       telemetry
       SLA

Controller ──► Edge A
       policy
       routes
       tunnel information
       configuration
Data plane
Host A
  │
  ▼
SD-WAN Edge A
  │
  │ encrypted overlay
  ▼
Internet/MPLS/5G
  │
  ▼
SD-WAN Edge B
  │
  ▼
Host B

The controller is not in the packet forwarding path.

That gives SD-WAN scalability.

5. How does routing integrate with the control plane?

This is where you should connect SD-WAN with your previous BGP/ECMP discussion.

For example:

                 SD-WAN Controller
                         │
                  Policy / topology
                         │
          ┌──────────────┴──────────────┐
          │                             │
      SD-WAN Edge A                SD-WAN Edge B
          │                             │
          │                             │
       BGP/OSPF                     BGP/OSPF
          │                             │
       Underlay                       Underlay
          │                             │
       Internet/MPLS/5G fabric

There are actually two routing domains:

Underlay

Responsible for getting packets between SD-WAN tunnel endpoints.

BGP / OSPF / IS-IS
        ↓
Underlay reachability
        ↓
IP connectivity
Overlay

SD-WAN control plane decides:

Which branch?
Which tunnel?
Which transport?
Which policy?
Which path?

So:

Underlay provides IP reachability; SD-WAN overlay control plane determines how applications should use that connectivity.

6. Where does BGP fit?

This is a common Staff Engineer interview trap.

Don't say:

"SD-WAN controller replaces BGP."

Instead:

             SD-WAN CONTROL PLANE
                      │
             Overlay policy/routes
                      │
                      ▼
              SD-WAN Edge
                 │       │
              BGP       BGP
                 │       │
                 ▼       ▼
              LAN      WAN

BGP can still be used for:

LAN route exchange
data-center connectivity
cloud connectivity
WAN underlay
Internet connectivity

The SD-WAN controller can orchestrate the overlay while conventional routing protocols continue to operate where appropriate.

7. How would you integrate an external NMS/OSS with SD-WAN?

This is probably the most relevant architecture if you're thinking about NMS + SD-WAN + agentic automation.

I would use:

                    ┌───────────────────────┐
                    │       NMS / OSS       │
                    │                       │
                    │ Inventory             │
                    │ Monitoring            │
                    │ Assurance             │
                    │ Automation             │
                    └───────────┬───────────┘
                                │
                         API / gRPC
                                │
                    ┌───────────▼───────────┐
                    │   SD-WAN Controller   │
                    │                       │
                    │ Policy                │
                    │ Topology              │
                    │ Routes                │
                    │ Device lifecycle      │
                    └───────────┬───────────┘
                                │
                         Controller protocol
                                │
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
          Edge A             Edge B             Edge C

The NMS should generally not directly manipulate every forwarding detail on every edge.

Instead:

NMS
 │
 │ "Create branch VPN"
 ▼
SD-WAN Controller
 │
 │ translates intent → device-specific config
 ▼
SD-WAN Edges

This gives you centralized intent and policy management.

8. In your Agentic AI architecture

You could integrate SD-WAN control plane very naturally with the architecture we've discussed earlier:

                 USER INTENT
                     │
                     ▼
              ┌─────────────┐
              │   OBSERVE   │
              │             │
              │ topology    │
              │ devices     │
              │ links       │
              │ SLA         │
              └──────┬──────┘
                     ▼
              ┌─────────────┐
              │    THINK    │
              │             │
              │ risk        │
              │ blast radius│
              │ path impact │
              │ policy      │
              └──────┬──────┘
                     ▼
              ┌─────────────┐
              │   ACTION    │
              │             │
              │ SD-WAN API  │
              │ Controller  │
              └──────┬──────┘
                     ▼
              SD-WAN Controller
                     │
             ┌───────┼────────┐
             ▼       ▼        ▼
           Edge A  Edge B   Edge C
                     │
                     ▼
               Actual traffic
                     │
                     ▼
                  OBSERVE

This creates a closed-loop control system:

Observe
   ↓
Analyze
   ↓
Decide
   ↓
Intent / Policy
   ↓
SD-WAN Controller
   ↓
Network
   ↓
Telemetry
   └──────────────► Observe
9. Example: automatic path optimization

Suppose:

Branch A ── MPLS ── Branch B
     │
     └── Internet ── Branch B

Current telemetry:

MPLS:
Latency = 80 ms
Loss    = 2%
Jitter  = 30 ms

Internet:
Latency = 25 ms
Loss    = 0.1%
Jitter  = 5 ms

Application requirement:

Voice:
Latency < 50 ms
Loss < 1%
Jitter < 10 ms

Agent/NMS detects SLA violation.

Instead of directly logging into Edge A:

Agent
  │
  ▼
SD-WAN Controller API
  │
  │ update application-aware policy
  ▼
Controller
  │
  ▼
Edge A
  │
  ▼
Select Internet tunnel

The actual forwarding then becomes:

Voice
  │
  ▼
Edge A
  │
  ├──── MPLS      X
  │
  └──── Internet ───────────► Edge B
10. The most important Staff-level distinction

I would explain the integration as:

SD-WAN control plane is the network's distributed decision-making layer. The controller maintains topology, policy and reachability state, while SD-WAN edges execute the forwarding decisions. An NMS or AI agent should integrate primarily through the controller's northbound APIs, converting high-level intent into controller policy rather than directly manipulating the data plane. The underlay continues to provide IP reachability using mechanisms such as BGP/OSPF, while the SD-WAN overlay provides application-aware path selection, segmentation and policy enforcement.

And if they ask "Where does telemetry fit?":

             ┌───────────────┐
             │ AI / NMS      │
             └───────▲───────┘
                     │
                  Telemetry
                     │
             ┌───────┴───────┐
             │ SD-WAN         │
             │ Controller     │
             └───────▲────────┘
                     │
              control/policy
                     │
             ┌───────┴───────┐
             │ SD-WAN Edges   │
             └───────┬────────┘
                     │
                  DATA PATH
                     │
                 WAN/LAN
Interview mental model

NMS/AI = Intent + Assurance
SD-WAN Controller = Control + Policy
BGP/OSPF = Underlay Routing
SD-WAN Edge = Enforcement + Forwarding
WAN links = Data Plane

That separation is the key to explaining SD-WAN control-plane integration at Staff Engineer level.
