# **Mitigating VLM Hallucination on Degraded Inputs: From Contrastive Decoding to Entropy-Weighted Soft Routing**

This report documents the architectural progression of mitigating hallucination in LLaVA-1.5 on severely degraded inputs (POPE benchmark). It tracks the failure of subtractive contrastive methods and establishes a single-pass, dynamic routing architecture that matches baseline discriminative power on clean data while preventing language-prior collapse on noisy data.

## **1\. The Illusion of VCD: Zero and Shuffle Hooks**

Visual Contrastive Decoding (VCD) attempts to mitigate hallucination by subtracting a "blinded" or degraded contrast prediction from the real prediction. To test the extreme limits of this hypothesis, two absolute hardware-level ablations were engineered via PyTorch forward hooks:

* **The Zero Hook:** Intercepts the vision-language projector output and applies torch.zeros\_like(), feeding the LLM 576 tokens of absolute zero.  
* **The Shuffle Hook:** Intercepts the vision embeddings and applies an on-GPU argsort to random noise, completely scrambling the spatial patch layout while preserving the mean token distribution.

### **The F1 vs. AUROC Paradox**

When evaluating standard VCD (and our extreme hook variants) on severely degraded images (e.g., Blur Level 4), the standard F1 score shows a massive apparent improvement (+20 points). However, this F1 gain is a mathematical illusion caused by a threshold shift, not a restoration of visual reasoning.

* **Degradation-Induced No-Bias:** As images degrade (Blur/Noise 1 \$\\rightarrow\$ 4), the VLM's precision stays high (\~90%) but recall collapses (\~70% \$\\rightarrow\$ \~24%). The model defaults to "No" because the visual signal is weak.  
* **The Yes-Ratio Inflation:** VCD subtracts the language prior. Because the language prior also heavily leans "No" on these inputs, subtracting it pushes the final logit distribution artificially toward "Yes".  
* **The AUROC Collapse:** AUROC measures threshold-independent discriminative power. Across all 27 degraded cells, VCD actually *lowers* the AUROC (e.g., dropping from the baseline 0.754 to 0.708 on Blur 4). Subtracting a contrast state from an already-weak visual signal flattens the logit distribution and destroys the forming visual features. VCD acts purely as a crude recalibration mechanism, hallucinating false positives to offset the degradation-induced false negatives.

## **2\. Background: Decoding by Contrasting Layers (DoLa)**

DoLa operates on the premise that factuality (or visual truth) crystallizes in the intermediate transformer layers, while the final layers (30–32) are dominated by language-modeling priors and structural formatting.

Standard DoLa dynamically selects an intermediate "premise" layer by calculating the Jensen-Shannon Divergence (JSD) between the intermediate layer's output distribution and the final layer's output distribution. It selects the layer with the highest JSD (the one that disagrees most with the final layer) and subtracts its logits.

### **Why Standard DoLa Fails on Degraded VLMs**

On highly blurred or noisy images, the final layers panic and hallucinate. Because the final layer's distribution is chaotic, the JSD calculation degenerates. The router constantly defaults to the earliest available layer (e.g., Layer 16). Forcing generation from Layer 16 causes severe grammar collapse, resulting in gibberish tokens, failed \<eos\> emission, and a 100% "unclear" rate on the POPE benchmark.

## **3\. Minimum Entropy Soft Routing**

To solve the subtractive signal-destruction of VCD and the degenerate routing of JSD, the architecture was shifted to **Minimum Entropy Routing**.

### **The Mechanism**

Instead of looking for divergence, the router looks for absolute confidence.

1. **Candidate Zone:** The hidden states of mid-to-late layers (18, 20, 22, 24, 26, 28, 30\) are passed through the final RMSNorm and LM Head.  
2. **Shannon Entropy:** For each candidate layer, the token-level Shannon Entropy \$H(P)\$ is calculated. Low entropy indicates a sharp, highly confident visual anchor.  
3. **Softmax Blending:** Instead of a brittle 1-of-N hard selection, the candidate logits are blended using a temperature-scaled softmax over their negative entropies. This ensures that if the visual truth is spread across Layers 22, 24, and 26, the evidence is additively fused rather than discarded.

**First-Token Hijack:** This routing is *only* applied to the very first token (the Yes/No decision). Control is then handed back to Layer 32 for the remaining 15 tokens to ensure perfect grammar and \<eos\> compliance.

### **Compute vs. Performance (The O(1) Advantage)**

Double-pass methods like standard VCD and the viewlayer baseline require O(2) inference latency (computing logits\_real and logits\_cd in parallel), dropping throughput from 0.08 seconds/question to 0.16 seconds/question and doubling VRAM usage.

Entropy-Weighted Soft Routing is an **O(1) single-pass architecture**. It requires no contrast image. By simply hunting the internal confidence peak, it achieves:

* **Clean Images (Level 1):** 0.9380 AUROC (perfectly matching the raw baseline, eliminating the early-exit penalty).  
* **Blur (Level 4):** 0.7582 AUROC (beating standard VCD's 0.708).  
* **Noise (Level 4):** 0.7379 AUROC (beating standard VCD's 0.708).

It achieves near-parity with the heavy viewlayer method (\~0.78 AUROC) while maintaining a 50% lower latency footprint.

## **4\. Future Architectural Directions**

To bridge the final 2-3 point AUROC gap on severe degradations while strictly maintaining the O(1) single-pass requirement, several extensions can be explored:

* **Dynamic Temperature Scaling:** Currently, the soft-routing softmax uses a static temperature (\$T=0.5\$). If the input image is heavily degraded, the baseline entropy across *all* layers rises. Implementing a dynamic \$T\$ that scales inversely with the mean entropy of the candidate block would force the softmax to harden on noisy images (isolating the single best visual anchor) while softening on clean images (allowing full-depth reasoning).  
* **Cross-Attention Cache Zeroing (The Hybrid Approach):** Hallucination occurs when Text Attention overrides Visual Attention. While Soft Routing cuts the depth of the forward pass, applying the earlier "Zero Hook" specifically to the text-token keys/values in the KV-cache would alter the breadth of the attention. A text-starved KV-cache combined with an entropy-weighted mid-layer exit would create a highly grounded, ultra-conservative VLM.  
* **Entropy-Gradient Early Exits:** Rather than evaluating a fixed block of candidate layers, the router could track the derivative of the entropy \$\\frac{dH}{dL}\$ layer by layer. The forward pass could be entirely short-circuited the moment the entropy gradient turns positive (indicating that the visual features are beginning to dissolve into language priors), saving both compute cycles and preventing hallucination.

