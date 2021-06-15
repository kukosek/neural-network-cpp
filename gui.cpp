#include <iostream>
#include <gtk/gtk.h>
#include <sstream>
#include <time.h>
#include <string>
#include <signal.h>
#include <vector>
#include "mnist/mnist_reader.hpp"
#include <Eigen/Dense>
#include "NeuralNetwork.hpp"
#include <omp.h>

using Eigen::VectorXd;

NeuralNetwork network;

GtkWidget *window;
GtkWidget *label;
const int pixelsize = 28;

std::string dirnameOf(const std::string& fname)
{
     size_t pos = fname.find_last_of("\\/");
     return (std::string::npos == pos)
         ? ""
         : fname.substr(0, pos);
}

bool network_ready = false;

static void print_hello (GtkWidget *widget, gpointer data) {
	GtkFileChooserNative *native;
	GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_OPEN;
	gint res;

	native = gtk_file_chooser_native_new ("Open model folder",
										  GTK_WINDOW(window),
										  action,
										  NULL,
										  NULL);

	res = gtk_native_dialog_run (GTK_NATIVE_DIALOG (native));
	if (res == GTK_RESPONSE_ACCEPT) {
		char *filename;
		GtkFileChooser *chooser = GTK_FILE_CHOOSER (native);
		filename = gtk_file_chooser_get_filename(chooser);
		std::string filenameStr = std::string(filename);
		std::string dirname = dirnameOf(filenameStr);
		std::cout << dirname << " -> ";
		g_free (filename);

		bool success = network.load_from_files(dirname);
		if (!success) {
			std::cout << "Failed loading input network" << std::endl;

			gtk_label_set_markup (GTK_LABEL (label), "<big>Failed loading model</big>");
		}else{
			std::cout << "Loaded model" << std::endl;
			gtk_label_set_markup (GTK_LABEL (label), "<big>Model loaded</big>");
			network_ready = true;
		}
	  }

	g_object_unref (native);
}


/* Surface to store current scribbles */
static cairo_surface_t *surface = NULL;

static void clear_surface (void)
{
	if (network_ready) {
		gtk_label_set_markup (GTK_LABEL (label), ("Cleared"));
		for (int i=0; i<pixelsize*pixelsize; i++) {
			network.layers[0].activations[i] = 0.0;
		}
	}
  cairo_t *cr;

  cr = cairo_create (surface);

  cairo_set_source_rgb (cr, 1, 1, 1);
  cairo_paint (cr);

  cairo_destroy (cr);
}

/* Create a new surface of the appropriate size to store our scribbles */
static gboolean configure_event_cb (GtkWidget         *widget,
                    GdkEventConfigure *event,
                    gpointer           data)
{
  if (surface)
    cairo_surface_destroy (surface);

  surface = gdk_window_create_similar_surface (gtk_widget_get_window (widget),
                                               CAIRO_CONTENT_COLOR,
                                               gtk_widget_get_allocated_width (widget),
                                               gtk_widget_get_allocated_height (widget));

  /* Initialize the surface to white */
  clear_surface ();

  /* We've handled the configure event, no need for further processing. */
  return TRUE;
}

/* Redraw the screen from the surface. Note that the ::draw
 * signal receives a ready-to-be-used cairo_t that is already
 * clipped to only draw the exposed areas of the widget
 */
static gboolean
draw_cb (GtkWidget *widget,
         cairo_t   *cr,
         gpointer   data)
{
  cairo_set_source_surface (cr, surface, 0, 0);
  cairo_paint (cr);

  return FALSE;
}


/* Draw a rectangle on the surface at the given position */
static void draw_brush (GtkWidget *widget, gdouble    x, gdouble    y) {
	if (network_ready) {
		cairo_t *cr;


		/* Paint to the surface, where we store our state */
		cr = cairo_create (surface);

		const float widget_height = gtk_widget_get_allocated_height(widget);
		const float widget_width = gtk_widget_get_allocated_width(widget);

		gint wx, wy;
		gtk_widget_translate_coordinates(widget, gtk_widget_get_toplevel(widget), 0, 0, &wx, &wy);

		const int brush_size = widget_width / pixelsize;

		int yx_size = brush_size;

		int pixel_x = (int)x / yx_size;
		int pixel_y =  (int)y/yx_size;


		x = pixel_x * yx_size;
		y = pixel_y * yx_size;

		const int pixel_idx = pixel_y*pixelsize + pixel_x;
		if (pixel_idx > 0 && pixel_idx < pixelsize*pixelsize) {
			network.layers[0].activations[pixel_idx] = 255.0;
		}
		pixel_x++;
		pixel_y++;

		network.calculate();

		// activations of the output layer
		VectorXd result_activations = network.layers[network.layers.size()-1].activations;

		// determine the brightest neuron
		// (thats the guess of the network, whats the digit)
		int max_idx = 0;
		double max_value = 0.0;
		for (int i=0; i<result_activations.size(); i++) {
			if (result_activations[i] > max_value) {
				max_idx = i;
				max_value = result_activations[i];
			}
		}

		gtk_label_set_markup (GTK_LABEL (label), ("I think it is a <big>"+std::to_string(max_idx)+"</big>").c_str());

		cairo_rectangle (cr, x, y, brush_size, brush_size);
		cairo_fill (cr);


		cairo_destroy (cr);

		/* Now invalidate the affected region of the drawing area. */
		gtk_widget_queue_draw_area (widget, x , y , brush_size, brush_size);
	}
}

/* Handle button press events by either drawing a rectangle
 * or clearing the surface, depending on which button was pressed.
 * The ::button-press signal handler receives a GdkEventButton
 * struct which contains this information.
 */
static gboolean
button_press_event_cb (GtkWidget      *widget,
                       GdkEventButton *event,
                       gpointer        data)
{
  /* paranoia check, in case we haven't gotten a configure event */
  if (surface == NULL)
    return FALSE;

  if (event->button == GDK_BUTTON_PRIMARY)
    {
      draw_brush (widget, event->x, event->y);
    }
  else if (event->button == GDK_BUTTON_SECONDARY)
    {
      clear_surface ();
      gtk_widget_queue_draw (widget);
    }

  /* We've handled the event, stop processing */
  return TRUE;
}

/* Handle motion events by continuing to draw if button 1 is
 * still held down. The ::motion-notify signal handler receives
 * a GdkEventMotion struct which contains this information.
 */
static gboolean
motion_notify_event_cb (GtkWidget      *widget,
                        GdkEventMotion *event,
                        gpointer        data)
{
  /* paranoia check, in case we haven't gotten a configure event */
  if (surface == NULL)
    return FALSE;

  if (event->state & GDK_BUTTON1_MASK)
    draw_brush (widget, event->x, event->y);

  /* We've handled it, stop processing */
  return TRUE;
}

static void
close_window (void)
{
  if (surface)
    cairo_surface_destroy (surface);
}



static void activate (GtkApplication *app, gpointer user_data) {
	GtkWidget *grid;
	GtkWidget *button;
	GtkWidget *button_box;
	GtkWidget *drawing_area;


	window = gtk_application_window_new (app);
	gtk_window_set_title (GTK_WINDOW (window), "Neural network digit recognition");
	gtk_window_set_default_size (GTK_WINDOW (window), 200, 200);
	gtk_container_set_border_width (GTK_CONTAINER (window), 10);

	grid = gtk_grid_new ();
	gtk_container_add (GTK_CONTAINER (window), grid);

	button_box = gtk_button_box_new (GTK_ORIENTATION_HORIZONTAL);
	gtk_grid_attach (GTK_GRID (grid), button_box, 0, 0, 1, 1);

	button = gtk_button_new_with_label ("Select model folder");
	g_signal_connect (button, "clicked", G_CALLBACK (print_hello), NULL);
	gtk_container_add (GTK_CONTAINER (button_box), button);

	drawing_area = gtk_drawing_area_new ();
	/* set a minimum size */
	gtk_widget_set_size_request (drawing_area, 300, 300);


	gtk_grid_attach (GTK_GRID (grid), drawing_area, 0, 1, 2, 1);

	/* Signals used to handle the backing surface */
	g_signal_connect (drawing_area, "draw",
					G_CALLBACK (draw_cb), NULL);
	g_signal_connect (drawing_area,"configure-event",
					G_CALLBACK (configure_event_cb), NULL);

	/* Event signals */
	g_signal_connect (drawing_area, "motion-notify-event",
					G_CALLBACK (motion_notify_event_cb), NULL);
	g_signal_connect (drawing_area, "button-press-event",
					G_CALLBACK (button_press_event_cb), NULL);

	/* Ask to receive events the drawing area doesn't normally
	* subscribe to. In particular, we need to ask for the
	* button press and motion notify events that want to handle.
	*/
	gtk_widget_set_events (drawing_area, gtk_widget_get_events (drawing_area)
									 | GDK_BUTTON_PRESS_MASK
									 | GDK_POINTER_MOTION_MASK);
	GtkTextBuffer *buffer;

	label = gtk_label_new (NULL);
	gtk_label_set_markup (GTK_LABEL (label), "<big>Please load model</big>");


	gtk_grid_attach (GTK_GRID (grid), label, 0, 2, 2, 1);


	gtk_widget_show_all (window);
}


void signal_handler(int s) {
	exit(1);
}

void print_usage() {
	std::cout << "USAGE:" << std::endl
	<<  "	--input-dir,  -i string			sets training data dir to load" << std::endl;
	exit(1);
}

int main(int argc, char* argv[]) {
	std::string inputfolder_name ;
	bool layersizes_set = false;
	struct sigaction sigIntHandler;

	sigIntHandler.sa_handler = signal_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;

	sigaction(SIGINT, &sigIntHandler, NULL);


    // MNIST_DATA_LOCATION set by MNIST cmake config std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

	// setup the network
	network = NeuralNetwork();

	bool skip_next_training = false;


	GtkApplication *app;
	int status;
	app = gtk_application_new ("org.gtk.example", G_APPLICATION_FLAGS_NONE);
	g_signal_connect (app, "activate", G_CALLBACK (activate), NULL);
	status = g_application_run (G_APPLICATION (app), argc, argv);
	g_object_unref (app);

	return status;


	std::cout << "Starting testing iterations" << std::endl << std::endl;

	while (true) {
		const int desired_number_scalar = 255;


		int correct_tests = 0;
		int bad_tests = 0;
		double cost_sum = 0.0;
		// for every image in test dataset
		for (int image_idx=0; image_idx < dataset.test_labels.size(); image_idx++) {
			// for every pixel of image
			for (int i=0; i<dataset.test_images[image_idx].size(); i++) {
				double value = dataset.test_images[image_idx][i] / 255.0;
				network.layers[0].activations[i] = value;
			}

			// calculate activations of neuron
			network.calculate();


			// can be used for cost calculations
			VectorXd desired = VectorXd::Constant(10, 0.0);
			int target_number = dataset.test_labels[image_idx];
			desired[target_number] = desired_number_scalar;

			cost_sum += network.get_cost(desired);

			// activations of the output layer
			VectorXd result_activations = network.layers[network.layers.size()-1].activations;

			// determine the brightest neuron
			// (thats the guess of the network, whats the digit)
			int max_idx = 0;
			double max_value = 0.0;
			for (int i=0; i<result_activations.size(); i++) {
				if (result_activations[i] > max_value) {
					max_idx = i;
					max_value = result_activations[i];
				}
			}

			// increase stat counters
			if (max_idx == target_number) correct_tests++;
			else bad_tests++;

			// detailed test prints
			// const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, "\t", " ", "", "", "", "");
			// std::cout << target_number << " cost " << network.get_cost(desired) << " output: " << network.layers[network.layers.size()-1].activations.format(fmt) << std::endl;
		}

		unsigned int n_of_tests = (correct_tests+bad_tests);
		double network_cost = cost_sum / n_of_tests;
		double network_sucess_rate = (double)correct_tests / n_of_tests;
		std::cout << "    of train "<<
			"; Last correct: " << correct_tests << "/" << n_of_tests
			<< "\t\r" << std::flush;

		if (skip_next_training) {
			skip_next_training = false;
		}else{
			//network.randomize();
		}
	}

    return 0;
}
