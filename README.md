
# Molecular Dynamics Simulation (2D)

This project simulates 2D molecular dynamics of particles in a box using Lennard-Jones interactions and Verlet integration.  
It includes both a text output and an animation of the particle motion.

## Animation Example

You can generate and view the animation as a GIF (compatible everywhere):

![Simulation Animation](particle_animation.gif)

## How to Save the Animation

The animation can be saved as a GIF file using the code below (works on any platform with Python and pillow):

```python
ani.save("particle_animation.gif", writer="pillow")

