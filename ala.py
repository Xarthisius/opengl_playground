import numpy as np
# PyQt4 imports
from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
# PyOpenGL imports
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
from PIL import Image

# Window creation function.
def create_window(window_class):
    """Create a Qt window in Python, or interactively in IPython with Qt GUI
    event loop integration:
        # in ~/.ipython/ipython_config.py
        c.TerminalIPythonApp.gui = 'qt'
        c.TerminalIPythonApp.pylab = 'qt'
    See also:
        http://ipython.org/ipython-doc/dev/interactive/qtconsole.html#qt-and-the-qtconsole
    """
    app_created = False
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtGui.QApplication(sys.argv)
        app_created = True
    app.references = set()
    window = window_class()
    app.references.add(window)
    window.show()
    if app_created:
        app.exec_()
    return window

def compile_vertex_shader(source):
    """Compile a vertex shader from source."""
    vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    gl.glShaderSource(vertex_shader, source)
    gl.glCompileShader(vertex_shader)
    # check compilation error
    result = gl.glGetShaderiv(vertex_shader, gl.GL_COMPILE_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetShaderInfoLog(vertex_shader))
    return vertex_shader

def compile_fragment_shader(source):
    """Compile a fragment shader from source."""
    fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(fragment_shader, source)
    gl.glCompileShader(fragment_shader)
    # check compilation error
    result = gl.glGetShaderiv(fragment_shader, gl.GL_COMPILE_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetShaderInfoLog(fragment_shader))
    return fragment_shader

def link_shader_program(vertex_shader, fragment_shader):
    """Create a shader program with from compiled shaders."""
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)
    # check linking error
    result = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetProgramInfoLog(program))
    return program

# Vertex shader
VS = """
#version 330
// Attribute variable that contains coordinates of the vertices.
layout(location = 0) in vec2 position;

// Main function, which needs to set `gl_Position`.
void main()
{
    // The final position is transformed from a null signal to a sinewave here.
    // We pass the position to gl_Position, by converting it into
    // a 4D vector. The last coordinate should be 0 when rendering 2D figures.
    gl_Position = vec4(position.x, .2 * sin(20 * position.x), 0., 1.);
}
"""

# Fragment shader
FS = """
#version 330
// Output variable of the fragment shader, which is a 4D vector containing the
// RGBA components of the pixel color.
out vec4 out_color;

// Main fragment shader function.
void main()
{
    // We simply set the pixel color to yellow.
    out_color = vec4(1., 1., 0., 1.);
}
"""

# Vertex shader2
VS2 = """
#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition_modelspace;

// Output data ; will be interpolated for each fragment.
out vec2 UV;

void main()
{
    gl_Position = vec4(vertexPosition_modelspace,1);
    UV = (vertexPosition_modelspace.xy+vec2(1,1))/2.0;
}
"""

# Fragment shader2
FS2 = """
#version 330 core

in vec2 UV;

out vec3 color;

uniform sampler2D renderedTexture;
uniform float time;

void main(){
   color = texture( renderedTexture, UV + 0.005*vec2( sin(1024.0*UV.x),cos(768.0*UV.y)) ).xyz;
}
"""

class GLPlotWidget(QGLWidget):
    # default window size
    width, height = 600, 600

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""
        # background color
        gl.glClearColor(0, 0, 0, 0)
        # create a Vertex Buffer Object with the specified data
        self.vbo = glvbo.VBO(self.data)
        # compile the vertex shader
        vs = compile_vertex_shader(VS)
        # compile the fragment shader
        fs = compile_fragment_shader(FS)
        # compile the vertex shader
        self.shaders_program = link_shader_program(vs, fs)
        vs2 = compile_vertex_shader(VS2)
        fs2 = compile_fragment_shader(FS2)
        self.my_shaders_program = link_shader_program(vs2, fs2)

    def paintGL(self):
        """Paint the scene."""
        # clear the color and depth buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # initialize FrameBuffer
        fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)

        depthbuffer = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depthbuffer)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24,
                                 self.width, self.height)
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER,
            depthbuffer
        )
        # --- end FB init

        # generate the texture we render to, and set parameters
        renderedTexture = gl.glGenTextures(1)   # create target texture
        # bind to new texture, all future texture functions will modify this
        # particular one
        gl.glBindTexture(gl.GL_TEXTURE_2D, renderedTexture)
        # set how our texture behaves on x,y boundaries
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        # set how our texture is filtered
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)

        # occupy width x height texture memory, (None at the end == empty
        # image)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.width,
                        self.height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_INT, None)

        # --- end texture init

        # Set "renderedTexture" as our colour attachement #0
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D,
            renderedTexture,
            0 # mipmap level, normally 0
        )

        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        assert status == gl.GL_FRAMEBUFFER_COMPLETE, status

        # bind to fb
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
        # viewport is shared with the main context
        gl.glViewport(0, 0, self.width, self.height)
        #  Clear the screen
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # --- at this point everything should be set, draw away

        # THIS IS DRAWING PART
        # bind the VBO
        self.vbo.bind()
        # tell OpenGL that the VBO contains an array of vertices
        # prepare the shader
        gl.glEnableVertexAttribArray(0)
        # these vertices contain 2 single precision coordinates
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glUseProgram(self.shaders_program)
        # draw "count" points from the VBO
        gl.glDrawArrays(gl.GL_LINE_STRIP, 0, len(self.data))
        # END OF DRAWING PART

        debug_buffer = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGB,
                                       gl.GL_UNSIGNED_BYTE)
        image = Image.frombytes(mode="RGB", size=(self.width, self.height),
                                data=debug_buffer)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save("/tmp/foo.jpg")

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0) # unbind FB
        gl.glBindTexture(gl.GL_TEXTURE_2D, renderedTexture) # bind to result ??

        # THIS IS DRAWING PART II
        # vertex arrays must be enabled using glEnableClientState
        gl.glEnable(gl.GL_TEXTURE_2D)  # ???

        texId = gl.glGetUniformLocation(self.my_shaders_program, "renderedTexture")

        gl.glUseProgram(self.my_shaders_program)
        # HOW TO DRAW TEXTURE
        gl.glActiveTexture(gl.GL_TEXTURE0);
        gl.glBindTexture(gl.GL_TEXTURE_2D, renderedTexture) # bind to result ??
        # Set our "renderedTexture" sampler to user Texture Unit 0
        gl.glUniform1i(texId, 0);
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex3f(-1.0,-1.0, 0.0)
        gl.glVertex3f( 1.0,-1.0, 0.0)
        gl.glVertex3f( 1.0, 1.0, 0.0)
        gl.glVertex3f(-1.0, 1.0, 0.0)
        gl.glEnd()
        #gl.glDrawElements(gl.GL_QUADS, 1, gl.GL_UNSIGNED_SHORT, indices)

        # gl.glDrawArrays(gl.GL_LINE_STRIP, 0, len(self.data))
        # END OF DRAWING PART II


    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport."""
        # update the window size
        self.width, self.height = width, height
        # paint within the whole window
        gl.glViewport(0, 0, width, height)

if __name__ == '__main__':
    # import numpy for generating random data points
    import sys
    import numpy as np

    # null signal
    data = np.zeros((10000, 2), dtype=np.float32)
    data[:,0] = np.linspace(-1., 1., len(data))

    # define a Qt window with an OpenGL widget inside it
    class TestWindow(QtGui.QMainWindow):
        def __init__(self):
            super(TestWindow, self).__init__()
            # initialize the GL widget
            self.widget = GLPlotWidget()
            self.widget.data = data
            # put the window at the screen position (100, 100)
            self.setGeometry(100, 100, self.widget.width, self.widget.height)
            self.setCentralWidget(self.widget)
            self.show()

    # show the window
    win = create_window(TestWindow)
