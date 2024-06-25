# ------------------------------------------------------
# Uses pygame to create a TSP solution window that can
# be kitted out with changing text and buttons.
# ------------------------------------------------------
from .common import Point
import pygame


class Button:
    def __init__(self, rect, text, callback, color=(255, 255, 255), hover_color=(200, 200, 200), font_color=(0, 0, 0),
                 visualizer=None):
        self.rect = rect
        self.text = text
        self.callback = callback
        self.color = color
        self.hover_color = hover_color
        self.font_color = font_color
        self.is_hovered = False

        self.visualizer = visualizer

    def draw(self, screen, font):
        current_color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, current_color, self.rect)
        text_surface = font.render(self.text, True, self.font_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

        # Mark the button area as dirty
        self.visualizer.mark_dirty(self.rect)

    def update(self, event_list):
        previously_hovered = self.is_hovered
        self.is_hovered = self.rect.collidepoint(pygame.mouse.get_pos())

        # If the hover state changes, mark the button as dirty
        if previously_hovered != self.is_hovered:
            self.visualizer.mark_dirty(self.rect)

        for event in event_list:
            if event.type == pygame.MOUSEBUTTONDOWN and self.is_hovered:
                self.callback(self)


class DynamicText:
    def __init__(self, position, initial_text, font, color=(255, 255, 255), visualizer=None):
        self.position = position
        self.text = initial_text
        self.font = font
        self.color = color
        self.visualizer = visualizer

        self.last_rendered_text = None
        self.last_rendered_rect = None

    def update_text(self, new_text):
        """
        Updates the currently displayed text.

        Arguments:
            new_text -- can be anything convertible into a string
        """
        # Update the text only if it has changed
        if new_text != self.last_rendered_text:
            self.text = str(new_text)
            self.redraw_text()

    def redraw_text(self):
        # Calculate the rect of the text
        text_rect = pygame.Rect(self.position, self.font.size(self.text))

        # Clear previous text rendering if any
        if self.last_rendered_rect:
            self.visualizer.mark_dirty(self.last_rendered_rect)

        # Mark the new text area as dirty
        self.visualizer.mark_dirty(text_rect)

        # Keep track of the last rendered text and its rect
        self.last_rendered_text = self.text
        self.last_rendered_rect = text_rect

    def draw(self, screen):
        # Create a separate surface for the text
        text_surface = self.font.render(self.text, True, self.color)

        # Blit the text surface onto the main screen
        screen.blit(text_surface, self.position)


class TSPVisualizer:
    def __init__(self, points: list, stop_callback: callable, hard_stop_callback: callable, input_callback: callable,
                 input_start, width: int = 800, height: int = 600, title: str = 'Traveling Salesman Problem'):
        pygame.init()
        self.points = points
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF | pygame.HWSURFACE)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)
        self.buttons = []
        self.dynamic_texts = []
        self.current_path = []
        self.dirty_rects = []

        # Create a background surface
        self.background = pygame.Surface((self.width, self.height))
        self.background.fill((25, 25, 25))

        self.stop_callback = stop_callback
        self.hard_stop_callback = hard_stop_callback
        self.input_callback = input_callback

        self.input = str(input_start) or ""
        self.running = False

    def add_dynamic_text(self, text: str, position: tuple) -> DynamicText:
        """
        Automatically creates a dynamic text object, integrates it into the visualizer and returns it.

        Arguments:
            text -- initial text to display
            position -- tuple with x and y
        """
        dynamic_text = DynamicText(position, text, self.font, visualizer=self)
        self.dynamic_texts.append(dynamic_text)
        self.mark_dirty(pygame.Rect(dynamic_text.position, dynamic_text.font.size(dynamic_text.text)))
        return dynamic_text  # Returns the object so the content can be manipulated later

    def add_button(self, rect: tuple, text: str, callback: callable) -> Button:
        """
        Automatically creates a button object, integrates it into the visualizer and returns it.

        Arguments:
            rect -- Four integers, x, y, width, height
            text -- initial text to display
            callback -- the function the button should call when being pressed (passes the button as the first argument)
        """
        button = Button(pygame.Rect(*rect), text, callback, visualizer=self)
        self.buttons.append(button)
        self.mark_dirty(button.rect)
        return button  # Returns the object so the content can be manipulated later

    def replace_points(self, new_points: list):
        """
        Replaces the current points, this triggers a full redraw, so don't do it unnecessarily often.

        Arguments:
            new_points -- list of Point objects or similar
        """
        self.points = new_points
        self.set_current_path([])  # Clears the old path
        self.dirty_rects = []  # Clear dirty rects and force full redraw
        self.mark_entire_screen_dirty()  # Method to redraw the whole screen

    def mark_dirty(self, rect):
        if rect not in self.dirty_rects:
            self.dirty_rects.append(rect)

    def set_current_path(self, path: list):
        """
        Update the current path to be drawn.

        Arguments:
            path -- A list of Point objects, has to be a subset of the current points, otherwise it won't be displayed.
        """
        self.current_path = path
        self.dirty_rects = []  # Clear dirty rects and force full redraw
        self.mark_entire_screen_dirty()  # Method to redraw the whole screen

    def hard_stop(self, _=None):
        """
        Stops the visualizer, uninitializes all pygame modules and calls the hard stop callback.

        Arguments:
            _ -- an ignored passes button that defaults to None
        """
        self.running = False
        pygame.quit()
        self.hard_stop_callback()

    def stop(self):
        """
        Stops the visualizer and calls the stop callback.
        """
        self.running = False
        self.stop_callback()

    def handle_events(self):
        need_redraw = False
        for event in pygame.event.get():  # Somehow writing this to a variable gets rid of most events
            if event.type == pygame.QUIT:
                self.hard_stop()
                return
            elif event.type == pygame.VIDEORESIZE or (event.type == pygame.ACTIVEEVENT and event.gain == 1):
                need_redraw = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for button in self.buttons:
                    if button.rect.collidepoint(event.pos):
                        button.callback(button)
            elif event.type == pygame.KEYDOWN and event.unicode.isdigit():
                self.input = (self.input + str(event.unicode))[-2:]
                self.input_callback()
        event_list = pygame.event.get()  # Could reuse the first event.get call for the button
        for button in self.buttons:  # update logic here, but if I do that closing the window stops working
            button.update(event_list)

        if need_redraw:
            self.mark_entire_screen_dirty()

    def run(self):
        """Starts the visualizer and is blocking."""
        self.running = True
        while self.running:
            self.handle_events()

            if self.dirty_rects:
                for rect in self.dirty_rects:
                    self.screen.blit(self.background, rect, rect)  # Redraw background in dirty rect
                    self.draw_path(rect)  # Redraw paths intersecting with rect
                    self.draw_points(rect)  # Redraw points intersecting with rect
                    self.draw_dynamic_texts(rect)  # Redraw texts intersecting with rect
                    self.draw_buttons(rect)  # Redraw buttons last
                pygame.display.update(self.dirty_rects)
                self.dirty_rects.clear()

            self.clock.tick(60)

    def draw_buttons(self, rect=None):
        for button in self.buttons:
            if rect is None or rect.colliderect(button.rect):
                button.draw(self.screen, self.font)

    def mark_entire_screen_dirty(self):
        screen_rect = pygame.Rect(0, 0, self.width, self.height)
        self.mark_dirty(screen_rect)

    def draw_dynamic_texts(self, rect=None):
        for dynamic_text in self.dynamic_texts:
            text_rect = pygame.Rect(dynamic_text.position, dynamic_text.font.size(dynamic_text.text))
            if rect is None or rect.colliderect(text_rect):
                dynamic_text.draw(self.screen)

    def draw_points(self, rect=None):
        for point in self.points:
            point_rect = pygame.Rect(point.x - 5, point.y - 5, 10, 10)  # Assuming points are drawn with a size of 10x10

            if rect is None or rect.colliderect(point_rect):
                pygame.draw.circle(self.screen,
                                   (235, 64, 52) if point != self.points[0] else (34, 139, 34),
                                   (point.x, point.y),
                                   max(1, 50 // len(self.points)))

    def point_exists(self, point):
        return point in self.points

    def draw_path(self, rect=None):
        for i in range(len(self.current_path)):
            if i + 1 < len(self.current_path):
                start_point = self.current_path[i]
                end_point = self.current_path[(i + 1)]

                if rect is not None:
                    # Check if the line segment intersects the rect
                    line_rect = pygame.Rect(start_point.x, start_point.y, end_point.x - start_point.x,
                                            end_point.y - start_point.y)
                    line_rect.normalize()  # Normalize the rectangle to have positive width and height
                    if not rect.colliderect(line_rect):
                        continue  # Skip drawing this line if it doesn't intersect the rect

                # Proceed with drawing the line segment
                pygame.draw.line(self.screen,
                                 (235, 64, 52),
                                 (start_point.x, start_point.y),
                                 (end_point.x, end_point.y), max(1, 20 // len(self.points)))


if __name__ == "__main__":
    # Example usage
    class TEST:
        running = False

        def __init__(self):
            self.text = None

        def set_text(self, text):
            self.text = text

        def run_toggle(self, button):
            button.text = "Stop" if button.text == "Run" else "Run"
            self.running = not self.running
            print("Running set to", self.running)
            self.text.update_text(self.running)

        @staticmethod
        def stop():
            print("Stopping ...")


    # Test class to demonstrate how to implement in a class
    test = TEST()

    # Points are static and can't be changed after the object is initialized
    example_points = [Point(100, 100), Point(200, 300), Point(300, 300), Point(500, 300)]
    # Give it a run callback function (what to call when the run or the stop button is pressed)
    # as well as the randomly generated points (cities)
    visualizer = TSPVisualizer(example_points, test.stop, test.stop, test.stop, test.stop)

    # How to add text
    current_length = visualizer.add_dynamic_text("HELL", (0, 0))
    # Giving the test class the text object, so it can change its contents like font color and displayed text
    test.set_text(current_length)

    # Displaying a path consisting only of the static points,
    # otherwise it'll trow an error (points don't have to be connected)
    example_path = [Point(300, 300), Point(200, 300), Point(100, 100)]
    visualizer.set_current_path(example_path)

    # How to add a button, we do not need to button object
    # as that is passed to the callback function (in this case visualizer.toggle_run)
    visualizer.add_button((100, 10, 100, 40), "Run", test.run_toggle)

    # Run the visualizer, preferably in another thread
    visualizer.run()
