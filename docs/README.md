<h1 align="center">
Differentiable Physics-based System Identification for 

Robotic Manipulation of Elastoplastic Materials
</h1>

<h2 align="center">
Code: <a href="https://github.com/IanYangChina/SI4RP-data"><img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="20" height="20"></a>
Video: <a href="https://www.youtube.com/watch?v=2-9JWRsQhTU"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/2560px-YouTube_full-color_icon_%282017%29.svg.png" width="25" height="20"></a>
</h2>

<p align="center">
  <img src="https://github.com/IanYangChina/SI4RP-data/blob/main/docs/Clay.gif" height="200"/>
  <img src="https://github.com/IanYangChina/SI4RP-data/blob/main/docs/Cloud_slime.gif" height="200"/>
</p>

<h2 align="center"> Abstract </h2>

### Robotic manipulation of volumetric elastoplastic deformable materials, from foods such as dough to construction materials like clay, is in its infancy, largely due to the difficulty of modelling and perception in a high-dimensional space. Simulating the dynamics of such materials is computationally expensive. It tends to suffer from inaccurately estimated physics parameters of the materials and the environment, impeding high-precision manipulation. Estimating such parameters from raw point clouds captured by optical cameras suffers further from heavy occlusions.
### To address this challenge, this work introduces a novel Differentiable Physics-based System Identification (DPSI) framework that enables a robot arm to infer the physics parameters of elastoplastic materials and the environment using simple manipulation motions and incomplete 3D point clouds, aligning the simulation with the real world.
### Extensive experiments show that with only a single real-world interaction, the estimated parameters, Young’s modulus, Poisson’s ratio, yield stress and friction coefficients, can accurately simulate visually and physically realistic deformation behaviours induced by unseen and long-horizon manipulation motions. Additionally, the DPSI framework inherently provides physically intuitive interpretations for the parameters in contrast to black-box approaches such as deep neural networks. 

<pre align="center">
  <img src="https://github.com/IanYangChina/SI4RP-data/blob/main/docs/real-platform-problem.png" width="700"/>


  <img src="https://github.com/IanYangChina/SI4RP-data/blob/main/docs/Diagram.png" width="710"/>





  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/Cardiff_University_%28logo%29.svg/512px-Cardiff_University_%28logo%29.svg.png" height="80"/>    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/UKRI_EPSR_Council-Logo_Horiz-RGB.png/799px-UKRI_EPSR_Council-Logo_Horiz-RGB.png" height="80"/>
</pre>

```bibtex
@article{yang2024differentiable,
  title={Differentiable Physics-based System Identification for Robotic Manipulation of Elastoplastic Materials},
  author={Yang, Xintong and Ji, Ze and Lai, Yu-Kun},
  journal={arXiv preprint arXiv:2411.00554},
  year={2024}
}
```
