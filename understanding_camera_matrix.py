import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D

fig, ax = plt.subplots()


plt.subplots_adjust(left=0.25, bottom=0.35)
t = [0.5,20]
a0 = 5
f0 = 3
delta_f = 5.0
s = [4,60]
l, = plt.plot(t, s, marker='o',markersize=10)
fig2=plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax.set_xlim([0,2200])
ax.set_ylim([0,1550])

X,Y,Z=[0,0,1],[0,0,1],[0,0.3,1]
a,=ax2.plot(X, Y, Z, "o", markersize=10)


axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
axamp = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
axZ = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axlen = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

#aspect ratio is the ratio of height to width
aspect_ratio=1550/2200
#diagonal fov is the angle between the rays of the corner of the image
d_fov=78
#we use the aspect ratio and the diagonal fov to calculate the horizontal and vertical fovs
xfov=2*(np.arctan(np.tan(d_fov*np.pi/180)*np.cos(np.arctan(aspect_ratio))))
yfov=2*(np.arctan(np.tan(d_fov*np.pi/180)*np.sin(np.arctan(aspect_ratio))))
#using these, we can find the maximum values for x and y given a maximum visible depth, zmax
zmax=1
xmax=np.sin(xfov)*zmax
ymax=np.sin(yfov)*zmax
#we can then put these into a scale matrix. Multiplying a point in xyz space by this matrix will get us that point in [-1,1],[-1,1],[0,1]
scale_mtx=np.array([[1/xmax,0,0],[0,1/ymax,0],[0,0,1/zmax]])
#we can invert this matrix to get us a matrix that undoes this operation
unscale_mtx=np.linalg.inv(scale_mtx)
#we then use this to transform the point 1,1,1 to the world coordinates. as a check, this should be equal to [xmax,ymax,zmax]
maxs=np.matmul(unscale_mtx,[1,1,1])
#we then set the graph limits to this so we can see the frame well
ax2.set_xlim([-maxs[0],maxs[0]])
ax2.set_ylim([-maxs[1],maxs[1]])
ax2.set_zlim([0,maxs[2]])

#we can then check to make sure our calculations worked by using trig to find the d_fov of our world system
print(maxs)
print('using cos',360/np.pi*np.arccos(maxs[-1]/(np.sqrt(maxs[0]**2+maxs[1]**2+maxs[2]**2))))
print('using sin',360/np.pi*np.arcsin((np.sqrt(maxs[0]**2+maxs[1]**2))/(np.sqrt(maxs[0]**2+maxs[1]**2+maxs[2]**2))))

#this section just sets up the graphs with lines to indicate the edges of the camera's vision
sequence=list(range(0,100))
for i in range(100):
    sequence[i]=i/100
sequence=np.array(sequence)
ax2.plot(sequence*xmax,sequence*ymax,sequence)
ax2.plot(-sequence*xmax,sequence*ymax,sequence)
ax2.plot(sequence*xmax,-sequence*ymax,sequence)
ax2.plot(-sequence*xmax,-sequence*ymax,sequence)
ax.margins(x=0)

#and these set up the sliders
sfreq = Slider(axfreq, 'Y', -1, 1.0, valinit=f0)
samp = Slider(axamp, 'X', -1, 1.0, valinit=a0)
sZ = Slider(axZ,'Z',0.1,1,valinit=2)
slen = Slider(axlen,'bar len', 0.01,0.5,valinit=0.05)

def update(val):
    amp = samp.val
    freq = sfreq.val
    Z=-sZ.val
    bar_len=slen.val
    #number of pixels in the x and y dimension
    rx=2200
    ry=1550
    if radio.value_selected=='World':
        #in this version, the x and y sliders correspond to the position on the world graph
        #this adds the other end of the bar
        extra_point=np.asarray([amp+bar_len,freq,Z])
        #this takes the bar's world coordinates and scales them back to normalized world coordinates
        norm_point=np.matmul(scale_mtx,extra_point)
        #this calculates the normalized image coordinates for the other end of the bar
        point_u=norm_point[0]/norm_point[2]
        point_v=norm_point[1]/norm_point[2]
        #this converts the normalized image coordinates to pixel coordinates
        pix_u,pix_v=(point_u+1)*rx/2,(point_v+1)*ry/2
        #this takes the controlled point in world coordinates and converts it to normalized world coordinates
        norm_controlled=np.matmul(scale_mtx,[amp,freq,Z])
        #this calculates the normalized image coordinates for the controlled part of the bar
        controlled_u=norm_controlled[0]/norm_controlled[2]
        controlled_v=norm_controlled[1]/norm_controlled[2]
        #this converts those normalized image coordinates to pixel coordinates
        u,v=(controlled_u+1)*rx/2,(controlled_v+1)*ry/2
        #these set the data to update the graph
        a.set_data([0,amp,extra_point[0]], [0,freq,extra_point[1]])
        update_val = np.asarray([0,-Z,-extra_point[2]])
        print(update_val)
        a.set_3d_properties(update_val)
        l.set_ydata([v,pix_v])
        l.set_xdata([u,pix_u])
    else:
        #in this version, the x and y sliders correspond to the position on the image
        #this updates those values from normalized (-1,1) to pixel coords
        upix=(amp+1)*rx/2
        vpix=(freq+1)*ry/2
        #this finds the x and y coordinates on a normalized world frame
        x=amp*Z
        y=freq*Z
        #this then takes those coordinates and scales them back to real world coordinates
        coord_list=np.matmul(unscale_mtx,[x,y,Z])
        x,y,Z=coord_list[0],coord_list[1],coord_list[2]
        #this adds the other end of the bar
        extra_point=np.asarray([x+bar_len,y,Z])
        #this takes the bar's world coordinates and scales them back to normalized world coordinates
        norm_point=np.matmul(scale_mtx,extra_point)
        #this calculates the normalized image coordinates for the other end of the bar
        point_u=norm_point[0]/norm_point[2]
        point_v=norm_point[1]/norm_point[2]
        #this converts the normalized image coordinates to pixel coordinates
        pix_u,pix_v=(point_u+1)*rx/2,(point_v+1)*ry/2
        #this sets the data to update the graph
        a.set_data([0,x,extra_point[0]], [0,y,extra_point[1]])
        update_val = np.asarray([0,-Z,-extra_point[-1]])
        print(type(update_val))
        a.set_3d_properties(update_val)
        l.set_ydata([vpix,pix_v])
        l.set_xdata([upix,pix_u])
    fig.canvas.draw_idle()    

sfreq.on_changed(update)
samp.on_changed(update)
sZ.on_changed(update)
slen.on_changed(update)
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('World', 'Camera'), active=0)

def colorfunc(label):
    if label=='Camera':
        world_coords=False
    else:
        world_coords=True
    print('coordinate system updated, change a slider')
    #l.set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(update)

plt.show()