#!/bin/env python3

# Python 2.7 compatibility
from __future__ import division

# GTK and GObject bindings
from gi.repository import Gtk, GObject

# Numpy
import numpy as np

# Delaunay triangulation
from scipy.spatial import Delaunay

# Various maths functions and constants
from math import pi, cos, sin, tan, exp, asin, acos, atan, log

# Path handling
import os.path

# Enable or disable debugging
DEBUG = False

# IPython for debugging
if DEBUG:
    import IPython

    
# Class for the fish
class Fish(object):
    radius, nearest, voronoi = range(0, 3)

    default_dt = 1/20

    default_pars = {"mu_u": 10, "sigma_u": 4.2,
                    "theta_u": 0.52, "theta_omega": 4.21,
                    "K_s": 2.5, "K_p": 3, "K_v": 12, "r_u": 4.2,
                    "r_omega": 2.4, "d_c": 30, "delta": 30,
                    "f_W_a": 5.26, "f_W_b": 0.13,
                    "f_c_a": 5, "f_c_b": 0.2,
                    "U_min": 1e-4, "U_max": 30, "Omega_max": 10}
    
    def __init__(self):
        self.ndim = 5
        self.nfish = 0
        # Initialise the state with no fish
        self.state = np.zeros((0, self.ndim))
        # Parameters for the simulation
        self.tank_radius = 50
        self.dt =  Fish.default_dt # Time step
        self.pars = dict(Fish.default_pars)

    @property
    def X(self):
        return self.state[:, :2]

    @property
    def x(self):
        return self.state[:, 0, None] # Avoid squeezing with None

    @property
    def y(self):
        return self.state[:, 1, None]

    @property
    def heading(self):
        return self.state[:, 2, None]

    @property
    def Omega(self):
        return self.state[:, 3, None]

    @property
    def U(self):
        return self.state[:, 4, None]

    def num_fish_update(self, fish_wanted):
        # Change the number of fish in the simulation
        if self.nfish > fish_wanted:
            self.state = np.delete(self.state, np.s_[fish_wanted:], 0)
        elif self.nfish < fish_wanted:
            self.state = np.append(self.state, np.zeros((fish_wanted - self.nfish, self.ndim)), 0)
            for i in range(self.nfish, fish_wanted):
                while True:
                    # Get x and y values within a circular tank
                    x = np.random.uniform(low=-self.tank_radius, high=self.tank_radius)
                    y = np.random.uniform(low=-self.tank_radius, high=self.tank_radius)
                    if (x**2 + y**2) < self.tank_radius**2:
                        break
                # Get a random heading direction
                heading = np.random.uniform(low=0, high=2*pi)
                # Add a fish with zero speed/turning speed
                self.state[i, :] = [x, y, heading, 0, self.pars["U_min"]]
        self.nfish = fish_wanted

    def update(self, method, radius, nearest, debug=False, grab=None):
        # Extract state variables (Nones are required to prevent dimension squeezing)
        X = self.state[:, :2] # cartesian position
        x = self.state[:, 0, None] # cartesian position x
        y = self.state[:, 1, None] # cartesian position y
        heading = self.state[:, 2, None] # heading angle
        Omega = self.state[:, 3, None] # turning speed
        U = self.state[:, 4, None] # forward speed

        # Easier access to the parameters
        p = self.pars

        # Calculate all-to-all distances
        x_dist = (x.T - x)
        y_dist = (y.T - y)
        self.dist = np.sqrt(x_dist**2 + y_dist**2)

        # Calculate adjacency matrices for all the different methods
        
        # Get everything with a certain radius
        self.radius_adj = self.dist < radius
        self.radius_adj[np.diag_indices(self.nfish)] = False
        # Get the nearest n neighbours
        idx = np.argsort(self.dist)[:, 1:nearest+1]
        self.nearest_adj = np.zeros((self.nfish, self.nfish), dtype=np.bool)
        self.nearest_adj[np.arange(0, self.nfish).reshape((self.nfish, 1)), idx] = True
        # Calculate the Voronoi tessellation
        if self.nfish > 3:
            # Construct the Delaunay triangulation
            tri = Delaunay(X) # , qhull_options="Qbb Qc Qz Qt") # options needed to make sure all vertices are left in
            # Construct an adjacency matrix from the triangulation
            self.voronoi_adj = np.zeros((self.nfish, self.nfish), dtype=np.bool)
            for simplex in tri.simplices:
                self.voronoi_adj[simplex[0], simplex[1]] = True
                self.voronoi_adj[simplex[0], simplex[2]] = True
                self.voronoi_adj[simplex[1], simplex[0]] = True
                self.voronoi_adj[simplex[1], simplex[2]] = True
                self.voronoi_adj[simplex[2], simplex[0]] = True
                self.voronoi_adj[simplex[2], simplex[1]] = True
        else:
            # All to all coupling if there aren't enough fish
            self.voronoi_adj = np.ones((self.nfish, self.nfish), dtype=np.bool)
        # Choose the correct adjacency matrix
        if method == Fish.radius:
            self.adj = self.radius_adj
        elif method == Fish.nearest:
            self.adj = self.nearest_adj
        else:
            self.adj = self.voronoi_adj

        # Some trig
        sin_heading = np.sin(heading)
        cos_heading = np.cos(heading)

        # Calculate the wall distance for each fish
        x_hx = x*cos_heading
        y_hy = y*sin_heading
        r2_xy = x**2 + y**2
        self.wall_dist = -(x_hx + y_hy) + np.sqrt((x_hx + y_hy)**2 - (r2_xy - self.tank_radius**2))
        self.wall_angle = np.arctan2(y + self.wall_dist*sin_heading,
                                     x + self.wall_dist*cos_heading) - heading
        self.wall_angle[self.wall_angle < -pi] += 2*pi

        # Mask adjacency matrix for distance
        adj = self.adj & (self.dist < p["d_c"])
        
        # Calculate the number of neighbours
        N_i = adj.sum(axis=1)
        N_i_idx = N_i != 0

        # Calculate the position angle of adjacent fish relative to the heading angle
        self.theta_ij = np.arctan2(y_dist[adj], x_dist[adj]) - np.repeat(heading, N_i) # only bother calculating for adjacent fish for speed

        # Helpers
        cos_theta = np.cos(self.theta_ij)
        sin_theta = np.sin(self.theta_ij)
        phi = heading.T - heading
        sin_phi = np.sin(phi[adj])
        
        # Calculate distance cut off function for adjacent fish
        f_d = 1 - np.exp((self.dist[adj] - p["d_c"])/p["delta"])
        f_d[f_d < 0] = 0

        # Calculate U_i_star
        U_star = np.zeros((self.nfish, self.nfish))
        U_star[adj] = f_d*p["K_s"]*(self.dist[adj] - p["r_u"])*cos_theta
        U_i_star = U_star.sum(axis=1)
        U_i_star[N_i_idx] /= (N_i[N_i_idx]*p["theta_u"])

        # Calculate Omega_i_star
        U_rep = np.repeat(U, N_i)
        Omega_star = np.zeros((self.nfish, self.nfish))
        Omega_star[adj] = f_d*(1 + cos_theta)*(p["K_p"]*(self.dist[adj] - p["r_omega"])*sin_theta/U_rep
                                                    + U_rep/p["mu_u"]*p["K_v"]*sin_phi/p["theta_omega"])
        Omega_i_star = Omega_star.sum(axis=1)
        Omega_i_star[N_i_idx] /= N_i[N_i_idx]
        
        # Calculate wall avoidance
        f_W = np.sign(self.wall_angle)*p["f_W_a"]*np.exp(-p["f_W_b"]*self.wall_dist)
        
        # Equations of motion
        dU_dt = -p["theta_u"]*(U - p["mu_u"] - U_i_star.reshape((self.nfish, 1)))
        dOmega_dt = -p["theta_omega"]*(Omega + f_W - Omega_i_star.reshape((self.nfish, 1)))

        # Coupling of the stochastic terms
        f_c = p["f_c_a"]*np.exp(-p["f_c_b"]*U);
        
        # Euler-Maruyama
        U_new = U + dU_dt*self.dt + np.random.randn(self.nfish, 1)*p["sigma_u"]*np.sqrt(self.dt)
        Omega_new = Omega + dOmega_dt*self.dt + np.random.randn(self.nfish, 1)*f_c*np.sqrt(self.dt)
        Omega_new = Omega_new

        # Check for bounds on speeds/turning speeds
        U_new[U_new < p["U_min"]] = p["U_min"]
        U_new[U_new > p["U_max"]] = p["U_max"]
        Omega_new[Omega_new < -p["Omega_max"]] = -p["Omega_max"]
        Omega_new[Omega_new > p["Omega_max"]] = p["Omega_max"]

        # Plain Euler
        heading_new = np.mod(heading + Omega*self.dt, 2*pi)
        x_new = x + U*cos_heading*self.dt
        y_new = y + U*sin_heading*self.dt

        # Check if any outside the tank
        r2_new = x_new**2 + y_new**2
        idx = r2_new > self.tank_radius**2
        x_new[idx] = x[idx]
        y_new[idx] = y[idx]
        U_new[idx] = p["U_min"]

        # Apply any grabs
        if grab is not None:
            x_new[grab, 0] = x[grab, 0]
            y_new[grab, 0] = y[grab, 0]
            heading_new[grab, 0] = heading[grab, 0]
            Omega_new[grab, 0] = Omega[grab, 0]
            U_new[grab, 0] = U[grab, 0]
        
        if debug:
            IPython.embed()
        
        # Update the state
        self.state[:, 0] = x_new[:, 0]
        self.state[:, 1] = y_new[:, 0]
        self.state[:, 2] = heading_new[:, 0]
        self.state[:, 3] = Omega_new[:, 0]
        self.state[:, 4] = U_new[:, 0]
        
        
# Handler for the different GUI events
class GUIHandler(object):
    def __init__(self, builder):
        # Initialise and store any information needed
        self.builder = builder
        for widget in ["zebrafish_demo_window",
                       "num_fish_range",
                       "radius_rad", "radius_range",
                       "nearest_rad", "nearest_range",
                       "voronoi_rad",
                       "interaction_area",
                       "show_interactions_chk",
                       "show_voronoi_chk",
                       "debug_update_chk",
                       "fish_area",
                       "speed_range",
                       "speed_interaction_range",
                       "speed_rand_range",
                       "turning_interaction_pos_range",
                       "turning_interaction_align_range",
                       "turning_rand_range",
                       "debug_btn"]:
            setattr(self, widget, builder.get_object(widget))

        # Timers to maintain interactivity
        self.pars_update_time = 100 # milliseconds
        self.pars_update_timer = None
        self.draw_time = 50 # milliseconds
        self.draw_timer = GObject.timeout_add(self.draw_time, self.update_fish)

        # Create some fish
        self.fish = Fish()

        # Update the parameters shown in the GUI
        self.on_reset_parameters_btn_clicked(None)
       
        # Scaling for plotting
        self.fish_plotting_scale = 1

        # For grabbing a fish
        self.grab = None
        self.cr_matrix = None
        
        # Attach the signals to this object
        builder.connect_signals(self)
        
        # Show the main window
        self.zebrafish_demo_window.show_all()

        # Remove debugging buttons if necessary
        if not DEBUG:
            self.debug_btn.set_visible(False)
            self.debug_update_chk.set_visible(False)

    def update_fish(self):
        # Check that the number of fish hasn't changed
        nfish = int(self.num_fish_range.get_value())
        if self.fish.nfish != nfish:
            self.fish.num_fish_update(nfish)
        # Update the fish simulation
        if self.radius_rad.get_active():
            method = Fish.radius
        elif self.nearest_rad.get_active():
            method = Fish.nearest
        else:
            method = Fish.voronoi
        self.fish.update(method,
                         radius=self.radius_range.get_value(),
                         nearest=round(self.nearest_range.get_value()),
                         debug=self.debug_update_chk.get_active(),
                         grab=self.grab)
        if self.debug_update_chk.get_active():
            self.debug_update_chk.set_active(False)
        # Draw the fish
        self.fish_area.queue_draw()
        return True

    def on_fish_area_draw(self, drawingarea, cr):
        if not hasattr(self.fish, "adj"):
            # The Fish.update function hasn't been run yet
            return

        # Whether to show interactions
        interactions = self.show_interactions_chk.get_active()

        # The fish
        fish_pos = self.fish.X
        fish_heading = self.fish.heading
        nfish = self.fish.nfish
        
        # Set up the transformations from tank space to visual space
        width = drawingarea.get_allocated_width()
        height = drawingarea.get_allocated_height()
        scale = 0.5*(min(width, height) - 30)/self.fish.tank_radius
        cr.set_source_rgb(0.4, 0.4, 0.4)
        cr.translate(width/2, height/2)
        cr.scale(scale, -scale)
        self.cr_matrix = cr.get_matrix()
        self.cr_matrix.invert()
        
        # Draw the tank
        cr.arc(0, 0, self.fish.tank_radius + 12/scale, 0, 2*pi)
        cr.set_line_width(1/scale) # Make sure that the line widths are in pixels
        cr.stroke()

        # Draw the Voronoi tessellation
        if self.show_voronoi_chk.get_active():
            cr.set_line_width(1/scale)
            cr.set_source_rgb(0, 0.6, 0)
            for i in range(0, nfish):
                for j in range(i + 1, nfish):
                    if self.fish.voronoi_adj[i, j]:
                        cr.move_to(fish_pos[i, 0], fish_pos[i, 1])
                        cr.line_to(fish_pos[j, 0], fish_pos[j, 1])
                        cr.stroke()

        # Draw the interaction radius if appropriate
        if self.radius_rad.get_active() and interactions:
            cr.arc(fish_pos[0, 0], fish_pos[0, 1], self.radius_range.get_value(), 0, 2*pi)
            cr.set_line_width(1/scale)
            cr.set_source_rgb(1, 0, 0)
            cr.stroke()
                        
        # Draw the fish
        sc = self.fish_plotting_scale
        for i in range(0, nfish):
            X = fish_pos[i, :]
            heading = fish_heading[i, 0]
            # Save the current transformation matrix
            cr.save()
            # Move from tank space to fish space
            cr.translate(X[0], X[1])
            cr.rotate(heading)
            # Scale to visual sizes
            cr.scale(1/scale, 1/scale)
            # Draw the fish
            cr.move_to(7*sc, 0)
            cr.line_to(0, 4*sc)
            cr.line_to(-12*sc, -1.5*sc)
            cr.line_to(-12*sc, 1.5*sc)
            cr.line_to(0, -4*sc)
            cr.close_path()
            cr.set_line_width(1)
            cr.set_source_rgb(200/255, 207/255, 229/255) # A nice light blue
            cr.fill_preserve()
            if i == 0:
                # Fish zero
                cr.set_source_rgb(0, 0, 1)
            elif self.fish.adj[0, i] and interactions:
                # Adjacent to fish zero
                cr.set_source_rgb(1, 0, 0)
            else:
                # Random fish
                cr.set_source_rgb(0, 0, 0)
            cr.stroke()
            # Restore the transformation matrix
            cr.restore()

    def on_zebrafish_demo_window_delete_event(self, *args):
        # Window has been closed - follow suit
        Gtk.main_quit(*args)

    def on_debug_btn_clicked(self, button):
        IPython.embed()

    def on_par_value_changed(self, range):
        # Put updates on a timer to avoid changing them thousands of times a second
        if self.pars_update_timer is None:
            self.pars_update_timer = GObject.timeout_add(self.pars_update_time, self.update_pars)

    def update_pars(self):
        self.pars_update_timer = None
        self.fish.pars["mu_u"] = self.speed_range.get_value()
        self.fish.pars["K_s"] = self.speed_interaction_range.get_value()
        self.fish.pars["sigma_u"] = self.speed_rand_range.get_value()
        self.fish.pars["K_p"] = self.turning_interaction_pos_range.get_value()
        self.fish.pars["K_v"] = self.turning_interaction_align_range.get_value()
        self.fish.pars["f_c_a"] = self.turning_rand_range.get_value()
        return False

    def on_reset_positions_btn_clicked(self, button):
        # Reset the fish positions
        nfish = self.fish.nfish
        self.fish.num_fish_update(0)
        self.fish.num_fish_update(nfish)

    def on_reset_parameters_btn_clicked(self, button):
        # Set the default parameters in the GUI
        self.speed_range.set_value(Fish.default_pars["mu_u"])
        self.speed_interaction_range.set_value(Fish.default_pars["K_s"])
        self.speed_rand_range.set_value(Fish.default_pars["sigma_u"])
        self.turning_interaction_pos_range.set_value(Fish.default_pars["K_p"])
        self.turning_interaction_align_range.set_value(Fish.default_pars["K_v"])
        self.turning_rand_range.set_value(Fish.default_pars["f_c_a"])
        self.fish.pars = dict(Fish.default_pars)

    def on_fish_area_button_press_event(self, widget, event):
        if self.cr_matrix is not None:
            pt = self.cr_matrix.transform_point(event.x, event.y)
            self.grab = np.argmin((self.fish.x - pt[0])**2 + (self.fish.y - pt[1])**2)

    def on_fish_area_button_release_event(self, widget, event):
        self.grab = None

    def on_fish_area_motion_notify_event(self, widget, event):
        if self.grab is not None:
            if self.cr_matrix is not None:
                pt = self.cr_matrix.transform_point(event.x, event.y)
                dist = np.sqrt(pt[0]**2 + pt[1]**2)
                if dist > (self.fish.tank_radius - 0.1):
                    angle = np.arctan2(pt[1], pt[0])
                    dist = self.fish.tank_radius - 0.1
                    pt = (dist*cos(angle), dist*sin(angle))
                self.fish.state[self.grab, 0] = pt[0]
                self.fish.state[self.grab, 1] = pt[1]

    
# Work out where the Python file is
py_path = os.path.dirname(os.path.realpath(__file__))
                
# Build the GUI
builder = Gtk.Builder()
builder.add_from_file(os.path.join(py_path, "zebrafish_demo.glade"))
gui = GUIHandler(builder)

# Start the main loop
Gtk.main()
