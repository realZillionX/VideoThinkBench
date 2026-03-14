# Prompt And Answer Audit

说明：
- 本文件列的是当前代码会生成的 prompt 和 answer 语义。
- 非 visual_puzzle 任务的 answer 用模板表示，因为具体值依赖随机采样。
- visual_puzzle 的 question / options / answer 也是模板，因为具体值同样依赖随机采样。

## angle_bisector
- group: eyeballing
- ti2v:
  A square white canvas shows two solid black line segments of medium thickness meeting at one shared vertex and opening into a clear angle, with no arrows and no extra marks. Five small candidate markers A-E sit near the hidden bisector on a short straight row: each marker is a small circle with white fill, a thin dark gray outline, and a black uppercase letter. The video holds this angle-and-candidates puzzle first, then draws one black bisector segment of medium thickness outward from the shared vertex through the interior of the angle toward the correct marker. In the final state, only the correct candidate changes to pale red fill with a dark red outline, while all other markers remain white. In portrait. Static camera.
- ti2i:
  A square white canvas shows two solid black line segments of medium thickness meeting at one shared vertex and opening into a clear angle, with no arrows and no extra marks. Five small candidate markers A-E sit near the hidden bisector on a short straight row: each marker is a small circle with white fill, a thin dark gray outline, and a black uppercase letter. The video holds this angle-and-candidates puzzle first, then draws one black bisector segment of medium thickness outward from the shared vertex through the interior of the angle toward the correct marker. In the final state, only the correct candidate changes to pale red fill with a dark red outline, while all other markers remain white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## arc_connect
- group: eyeballing
- ti2v:
  A square white canvas contains a wide vertical mask band in the center, about 35 percent of the canvas width, filled light gray and bounded by two medium-thickness medium gray edge lines. To the right of the mask are five thick dark gray arc fragments; each ends beside a candidate marker A-E drawn as a small white circle with a thin dark gray outline and a black uppercase letter, and only one option also has a matching left-side arc fragment in darker near-black. The video first holds the fully masked puzzle, then the central mask smoothly narrows and disappears so the hidden middle portions of the arcs are revealed, making exactly one arc read as one continuous left-to-right circle arc while the other four remain right-side arcs only. In the final revealed frame, the correct candidate alone switches to pale red fill with a dark red outline, while the other four markers stay white. In portrait. Static camera.
- ti2i:
  A square white canvas contains a wide vertical mask band in the center, about 35 percent of the canvas width, filled light gray and bounded by two medium-thickness medium gray edge lines. To the right of the mask are five thick dark gray arc fragments; each ends beside a candidate marker A-E drawn as a small white circle with a thin dark gray outline and a black uppercase letter, and only one option also has a matching left-side arc fragment in darker near-black. The video first holds the fully masked puzzle, then the central mask smoothly narrows and disappears so the hidden middle portions of the arcs are revealed, making exactly one arc read as one continuous left-to-right circle arc while the other four remain right-side arcs only. In the final revealed frame, the correct candidate alone switches to pale red fill with a dark red outline, while the other four markers stay white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## circle_center
- group: eyeballing
- ti2v:
  A square white canvas shows one large unfilled circle centered somewhere on the page, drawn with a thin black outline and no other geometry. Five candidate markers A-E are clustered around the hidden center of that circle; each marker is a small circle with white fill, a thin dark gray outline, and a black uppercase letter. The video begins with a short hold on the static circle and the five markers, and there is no construction line or extra shape added during the middle phase. The final state simply changes the exact center marker to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera.
- ti2i:
  A square white canvas shows one large unfilled circle centered somewhere on the page, drawn with a thin black outline and no other geometry. Five candidate markers A-E are clustered around the hidden center of that circle; each marker is a small circle with white fill, a thin dark gray outline, and a black uppercase letter. The video begins with a short hold on the static circle and the five markers, and there is no construction line or extra shape added during the middle phase. The final state simply changes the exact center marker to pale red fill with a dark red outline, while the other four markers remain white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## circle_tangent_line
- group: eyeballing
- ti2v:
  A square white canvas shows one large unfilled circle drawn with a black outline of medium thickness, plus one small hollow marker with white fill and a thick black outline sitting exactly on the circumference to mark the tangency point. Five candidate markers A-E form a short straight row near the hidden tangent direction; each is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the circle, the marked contact point, and the candidate row, then draws a single black tangent segment of medium thickness starting just outside that hollow tangency marker and extending in one direction toward the correct candidate. In the final frame, only the candidate on that tangent segment changes to pale red fill with a dark red outline, while the others remain white. In portrait. Static camera.
- ti2i:
  A square white canvas shows one large unfilled circle drawn with a black outline of medium thickness, plus one small hollow marker with white fill and a thick black outline sitting exactly on the circumference to mark the tangency point. Five candidate markers A-E form a short straight row near the hidden tangent direction; each is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the circle, the marked contact point, and the candidate row, then draws a single black tangent segment of medium thickness starting just outside that hollow tangency marker and extending in one direction toward the correct candidate. In the final frame, only the candidate on that tangent segment changes to pale red fill with a dark red outline, while the others remain white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## circle_tangent_point
- group: eyeballing
- ti2v:
  A square white canvas shows one large unfilled circle drawn with a black outline of medium thickness and one small hollow external marker with white fill and a thick black outline outside the circle. Exactly five candidate markers A-E lie directly on the circumference itself; each candidate is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the circle, the external marker, and the five circumference markers, then reveals the circle center as a small hollow white marker outlined in black, draws a black tangent segment from the external marker to the correct candidate, and finally draws a black radius from the center marker to that same candidate. In the final state, that tangency candidate alone changes to pale red fill with a dark red outline, while the other four circumference markers stay white. In portrait. Static camera.
- ti2i:
  A square white canvas shows one large unfilled circle drawn with a black outline of medium thickness and one small hollow external marker with white fill and a thick black outline outside the circle. Exactly five candidate markers A-E lie directly on the circumference itself; each candidate is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the circle, the external marker, and the five circumference markers, then reveals the circle center as a small hollow white marker outlined in black, draws a black tangent segment from the external marker to the correct candidate, and finally draws a black radius from the center marker to that same candidate. In the final state, that tangency candidate alone changes to pale red fill with a dark red outline, while the other four circumference markers stay white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## circumcenter
- group: eyeballing
- ti2v:
  A square white canvas shows a triangle drawn only as a black outline of medium thickness, with no fill and no center marks. Five candidate markers A-E sit near the hidden circumcenter; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the triangle and the candidate markers, then draws one black circumcircle of medium thickness centered on the correct option so that the circle passes exactly through all three triangle vertices. In the final state, the circumcenter marker alone changes to pale red fill with a dark red outline, while the other candidates remain white. In portrait. Static camera.
- ti2i:
  A square white canvas shows a triangle drawn only as a black outline of medium thickness, with no fill and no center marks. Five candidate markers A-E sit near the hidden circumcenter; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the triangle and the candidate markers, then draws one black circumcircle of medium thickness centered on the correct option so that the circle passes exactly through all three triangle vertices. In the final state, the circumcenter marker alone changes to pale red fill with a dark red outline, while the other candidates remain white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## color_grid
- group: visual_puzzle
- question:
  <QUESTION>
- options:
  [<OPTION_1>, <OPTION_2>, ...]
- answer:
  <OPTION_k>
- ti2v:
  Use the provided visual puzzle image as the starting frame. <QUESTION> A centered visual puzzle sits on a clean white square canvas. It shows nine evenly spaced pastel circles with thin black outlines arranged in a 3x3 grid, where the four corner circles share one color, the four edge-middle circles share a second color, and the center circle is a third color. One non-center position contains only a black question mark, which represents the missing circle color that should appear at that exact grid cell. The video holds on the puzzle frame, then smoothly crossfades to the solution so that only the question-mark region changes into the correct answer while every other shape stays fixed, and finally remains still on the solved image with a static camera and no zoom or pan.
- ti2i:
  Use the provided visual puzzle image as input. <QUESTION> Replace only the missing question-mark region with the correct answer while keeping every other shape unchanged. Output the solved image.
- ti2t:
  Use the provided visual puzzle image to solve the task. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. Answer with exactly one option.
- ti2t_answer:
  Answer: <OPTION_k>.
- ti2ti:
  Use the provided visual puzzle image as input. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. First determine the correct option, then replace only the missing question-mark region with that answer while keeping every other shape unchanged. Output the solved image.
- ti2ti_answer:
  {"text": "Answer: <OPTION_k>.", "image": "solution_image_path"}

## color_hexagon
- group: visual_puzzle
- question:
  <QUESTION>
- options:
  [<OPTION_1>, <OPTION_2>, ...]
- answer:
  <OPTION_k>
- ti2v:
  Use the provided visual puzzle image as the starting frame. <QUESTION> A centered regular hexagon appears on a clean white square canvas. The hexagon is divided from the center into six triangular wedges with thin black edges, and opposite wedges share the same pastel color chosen from blue, green, yellow, red, purple, or orange. One wedge is filled light gray and marked with a black question mark, indicating the missing wedge color. The video holds on the puzzle frame, then smoothly crossfades to the solution so that only the question-mark region changes into the correct answer while every other shape stays fixed, and finally remains still on the solved image with a static camera and no zoom or pan.
- ti2i:
  Use the provided visual puzzle image as input. <QUESTION> Replace only the missing question-mark region with the correct answer while keeping every other shape unchanged. Output the solved image.
- ti2t:
  Use the provided visual puzzle image to solve the task. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. Answer with exactly one option.
- ti2t_answer:
  Answer: <OPTION_k>.
- ti2ti:
  Use the provided visual puzzle image as input. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. First determine the correct option, then replace only the missing question-mark region with that answer while keeping every other shape unchanged. Output the solved image.
- ti2ti_answer:
  {"text": "Answer: <OPTION_k>.", "image": "solution_image_path"}

## color_overlap_squares
- group: visual_puzzle
- question:
  <QUESTION>
- options:
  [<OPTION_1>, <OPTION_2>, ...]
- answer:
  <OPTION_k>
- ti2v:
  Use the provided visual puzzle image as the starting frame. <QUESTION> A centered composition on a clean white square canvas shows three overlapping rotated squares with thin black outlines, and the whole cluster is slightly rotated. The three main squares use primary colors while the overlap regions show the corresponding mixed secondary colors, forming a compact layered arrangement of square and triangular regions. One side square is replaced by a light gray shape with a black question mark, indicating the missing square color. The video holds on the puzzle frame, then smoothly crossfades to the solution so that only the question-mark region changes into the correct answer while every other shape stays fixed, and finally remains still on the solved image with a static camera and no zoom or pan.
- ti2i:
  Use the provided visual puzzle image as input. <QUESTION> Replace only the missing question-mark region with the correct answer while keeping every other shape unchanged. Output the solved image.
- ti2t:
  Use the provided visual puzzle image to solve the task. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. Answer with exactly one option.
- ti2t_answer:
  Answer: <OPTION_k>.
- ti2ti:
  Use the provided visual puzzle image as input. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. First determine the correct option, then replace only the missing question-mark region with that answer while keeping every other shape unchanged. Output the solved image.
- ti2ti_answer:
  {"text": "Answer: <OPTION_k>.", "image": "solution_image_path"}

## color_size
- group: visual_puzzle
- question:
  <QUESTION>
- options:
  [<OPTION_1>, <OPTION_2>, ...]
- answer:
  <OPTION_k>
- ti2v:
  Use the provided visual puzzle image as the starting frame. <QUESTION> A large centered stack of four nested shapes sits on a clean white square canvas. All four shapes share the same geometry, either circles or regular polygons with black outlines, and they use four shades of one hue so the color changes steadily lighter or darker as the shapes shrink inward. The smallest inner shape is replaced by a light gray shape carrying a black question mark, indicating the missing final shade. The video holds on the puzzle frame, then smoothly crossfades to the solution so that only the question-mark region changes into the correct answer while every other shape stays fixed, and finally remains still on the solved image with a static camera and no zoom or pan.
- ti2i:
  Use the provided visual puzzle image as input. <QUESTION> Replace only the missing question-mark region with the correct answer while keeping every other shape unchanged. Output the solved image.
- ti2t:
  Use the provided visual puzzle image to solve the task. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. Answer with exactly one option.
- ti2t_answer:
  Answer: <OPTION_k>.
- ti2ti:
  Use the provided visual puzzle image as input. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. First determine the correct option, then replace only the missing question-mark region with that answer while keeping every other shape unchanged. Output the solved image.
- ti2ti_answer:
  {"text": "Answer: <OPTION_k>.", "image": "solution_image_path"}

## incenter
- group: eyeballing
- ti2v:
  A square white canvas shows a triangle drawn only as a black outline of medium thickness, with the interior left empty. Five candidate markers A-E are clustered near the hidden incenter; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the triangle and the candidate markers, then draws one black inscribed circle of medium thickness centered on the correct option so the circle sits inside the triangle and is tangent to all three sides. In the final state, only the incenter marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera.
- ti2i:
  A square white canvas shows a triangle drawn only as a black outline of medium thickness, with the interior left empty. Five candidate markers A-E are clustered near the hidden incenter; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the triangle and the candidate markers, then draws one black inscribed circle of medium thickness centered on the correct option so the circle sits inside the triangle and is tangent to all three sides. In the final state, only the incenter marker changes to pale red fill with a dark red outline, while the other four markers remain white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## isosceles_trapezoid
- group: eyeballing
- ti2v:
  A square white canvas shows three known vertices of an almost-complete isosceles trapezoid as an open black polyline of medium thickness with two connected segments already visible and one corner missing. Five candidate markers A-E are placed near that missing fourth vertex; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the open three-vertex polyline and the candidate markers, then draws two black segments of medium thickness from the visible right endpoint to the correct candidate and from that candidate to the upper-left visible vertex, completing a trapezoid whose two bases are parallel and whose two legs are equal. In the final frame, only the correct missing-corner marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera.
- ti2i:
  A square white canvas shows three known vertices of an almost-complete isosceles trapezoid as an open black polyline of medium thickness with two connected segments already visible and one corner missing. Five candidate markers A-E are placed near that missing fourth vertex; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the open three-vertex polyline and the candidate markers, then draws two black segments of medium thickness from the visible right endpoint to the correct candidate and from that candidate to the upper-left visible vertex, completing a trapezoid whose two bases are parallel and whose two legs are equal. In the final frame, only the correct missing-corner marker changes to pale red fill with a dark red outline, while the other four markers remain white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## maze_hexagon
- group: maze
- ti2v:
  A square-format video shows a hexagonal maze on a dark charcoal background. The maze is a centered honeycomb of light gray hexagonal walkable cells with thick black wall segments and rounded black wall caps, and the start and goal are solid red dots centered inside two different hexagons. The animation begins with the static maze, then a thick bright red path is progressively traced through the centers of edge-sharing hex cells without crossing any black wall edge. When the route is complete, the frame holds for one second on the finished path. In portrait. Static camera.
- ti2i:
  A square-format video shows a hexagonal maze on a dark charcoal background. The maze is a centered honeycomb of light gray hexagonal walkable cells with thick black wall segments and rounded black wall caps, and the start and goal are solid red dots centered inside two different hexagons. The animation begins with the static maze, then a thick bright red path is progressively traced through the centers of edge-sharing hex cells without crossing any black wall edge. When the route is complete, the frame holds for one second on the finished path.
- ti2t:
  Use the provided reasoning image to solve the task. Find a path connecting the two red dots without touching the black walls in the maze. Each traversable region has its ID printed on it. Present your answer as a list of IDs. Example: [1, 4, 3, 2].
- ti2t_answer:
  Answer: [<CELL_ID_1>, <CELL_ID_2>, ...].
- ti2ti:
  Find a path connecting the two red dots without touching the black walls in the maze. Each traversable region has its ID printed on it. Determine the correct path, then generate the solved maze image by drawing the red path and removing the cell ID numbers from the final image.
- ti2ti_answer:
  {"text": "Answer: [<CELL_ID_1>, <CELL_ID_2>, ...].", "image": "solution_image_path"}

## maze_labyrinth
- group: maze
- ti2v:
  A square-format video shows a circular labyrinth on a very dark charcoal background. The maze is made of pale gray concentric ring corridors and a pale gray central disk, separated by thick black circular walls and black radial spokes, with a solid red start dot centered inside the outer ring corridor and a solid red goal dot at the center. The animation begins with the static labyrinth, then a thick bright red path is progressively drawn as smooth arcs along ring-center lines and straight radial segments through shared openings, without crossing any black wall. After the line reaches the center, the completed route holds for one second. In portrait. Static camera.
- ti2i:
  A square-format video shows a circular labyrinth on a very dark charcoal background. The maze is made of pale gray concentric ring corridors and a pale gray central disk, separated by thick black circular walls and black radial spokes, with a solid red start dot centered inside the outer ring corridor and a solid red goal dot at the center. The animation begins with the static labyrinth, then a thick bright red path is progressively drawn as smooth arcs along ring-center lines and straight radial segments through shared openings, without crossing any black wall. After the line reaches the center, the completed route holds for one second.
- ti2t:
  Use the provided reasoning image to solve the task. Find a path connecting the two red dots without touching the black walls in the maze. Each traversable region has its ID printed on it. Present your answer as a list of IDs. Example: [1, 4, 3, 2].
- ti2t_answer:
  Answer: [<CELL_ID_1>, <CELL_ID_2>, ...].
- ti2ti:
  Find a path connecting the two red dots without touching the black walls in the maze. Each traversable region has its ID printed on it. Determine the correct path, then generate the solved maze image by drawing the red path and removing the cell ID numbers from the final image.
- ti2ti_answer:
  {"text": "Answer: [<CELL_ID_1>, <CELL_ID_2>, ...].", "image": "solution_image_path"}

## maze_square
- group: maze
- ti2v:
  A square-format video shows a square maze on a pure black background. The maze is a centered regular grid of square cells where walkable corridor cells are solid white, wall cells are solid black, and the start cell and goal cell are two solid red squares placed at opposite corners of the grid. The animation begins with the static maze, then a thick bright red square-cornered path with a flat square head is progressively drawn through the centers of side-adjacent white cells without entering any black wall cell. After the route reaches the goal, the completed red line holds for one second with the red endpoint cells still visible. In portrait. Static camera.
- ti2i:
  A square-format video shows a square maze on a pure black background. The maze is a centered regular grid of square cells where walkable corridor cells are solid white, wall cells are solid black, and the start cell and goal cell are two solid red squares placed at opposite corners of the grid. The animation begins with the static maze, then a thick bright red square-cornered path with a flat square head is progressively drawn through the centers of side-adjacent white cells without entering any black wall cell. After the route reaches the goal, the completed red line holds for one second with the red endpoint cells still visible.
- ti2t:
  Use the provided reasoning image to solve the task. Find a path connecting the two red dots without touching the black walls in the maze. Each traversable region has its ID printed on it. Present your answer as a list of IDs. Example: [1, 4, 3, 2].
- ti2t_answer:
  Answer: [<CELL_ID_1>, <CELL_ID_2>, ...].
- ti2ti:
  Find a path connecting the two red dots without touching the black walls in the maze. Each traversable region has its ID printed on it. Determine the correct path, then generate the solved maze image by drawing the red path and removing the cell ID numbers from the final image.
- ti2ti_answer:
  {"text": "Answer: [<CELL_ID_1>, <CELL_ID_2>, ...].", "image": "solution_image_path"}

## midpoint
- group: eyeballing
- ti2v:
  A square white canvas shows two large anchor circles at the endpoints of an invisible segment; each anchor is filled near-white and outlined in dark gray with a thick stroke. Near the hidden midpoint, five candidate markers A-E are shown as small white circles with thin dark gray outlines and black uppercase letters. The video first holds the two anchor circles and the five candidate markers, then draws one dark line segment of medium thickness directly between the two anchor centers while the anchor circles remain visibly on top of that segment. In the final state, only the exact midpoint marker changes to pale red fill with a dark red outline, while the other candidates stay white. In portrait. Static camera.
- ti2i:
  A square white canvas shows two large anchor circles at the endpoints of an invisible segment; each anchor is filled near-white and outlined in dark gray with a thick stroke. Near the hidden midpoint, five candidate markers A-E are shown as small white circles with thin dark gray outlines and black uppercase letters. The video first holds the two anchor circles and the five candidate markers, then draws one dark line segment of medium thickness directly between the two anchor centers while the anchor circles remain visibly on top of that segment. In the final state, only the exact midpoint marker changes to pale red fill with a dark red outline, while the other candidates stay white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## orthocenter
- group: eyeballing
- ti2v:
  A square white canvas shows a triangle drawn only as a black outline of medium thickness, with no helper lines and no interior labels. Five candidate markers A-E are placed near the hidden orthocenter; each is a small white circle with a thin dark gray outline and a black uppercase letter, and the cluster may lie inside or just outside the triangle. The video first holds the bare triangle and the candidate markers, then draws three black altitude segments of medium thickness that meet at one common point, using interior vertex-to-side drops for acute cases and extending the construction outside the triangle when the orthocenter lies outside. In the final state, only the orthocenter marker changes to pale red fill with a dark red outline, while the other candidates remain white. In portrait. Static camera.
- ti2i:
  A square white canvas shows a triangle drawn only as a black outline of medium thickness, with no helper lines and no interior labels. Five candidate markers A-E are placed near the hidden orthocenter; each is a small white circle with a thin dark gray outline and a black uppercase letter, and the cluster may lie inside or just outside the triangle. The video first holds the bare triangle and the candidate markers, then draws three black altitude segments of medium thickness that meet at one common point, using interior vertex-to-side drops for acute cases and extending the construction outside the triangle when the orthocenter lies outside. In the final state, only the orthocenter marker changes to pale red fill with a dark red outline, while the other candidates remain white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## parallel
- group: eyeballing
- ti2v:
  A square white canvas shows one black reference segment of medium thickness and one separate small hollow marker with white fill and a thick black outline marking the point that a new segment must pass through. Five candidate markers A-E sit near the hidden parallel direction on a short straight row; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the reference segment, the through-point marker, and the candidate row, then draws a single black segment of medium thickness starting just outside that hollow marker and extending toward the correct candidate in a direction parallel to the reference segment. In the final frame, only the candidate on that new parallel segment changes to pale red fill with a dark red outline, while the other markers remain white. In portrait. Static camera.
- ti2i:
  A square white canvas shows one black reference segment of medium thickness and one separate small hollow marker with white fill and a thick black outline marking the point that a new segment must pass through. Five candidate markers A-E sit near the hidden parallel direction on a short straight row; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the reference segment, the through-point marker, and the candidate row, then draws a single black segment of medium thickness starting just outside that hollow marker and extending toward the correct candidate in a direction parallel to the reference segment. In the final frame, only the candidate on that new parallel segment changes to pale red fill with a dark red outline, while the other markers remain white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## parallelogram
- group: eyeballing
- ti2v:
  A square white canvas shows three known vertices of a parallelogram as an open black broken line of medium thickness shaped like a V, with two adjacent sides already visible and one opposite corner missing. Five candidate markers A-E are placed near the missing corner; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the open broken line and the candidate markers, then draws two black closing segments of medium thickness from one visible endpoint to the correct candidate and from that candidate to the other visible endpoint so the full quadrilateral becomes a parallelogram. In the final state, only the correct missing-vertex marker changes to pale red fill with a dark red outline, while the other four candidates remain white. In portrait. Static camera.
- ti2i:
  A square white canvas shows three known vertices of a parallelogram as an open black broken line of medium thickness shaped like a V, with two adjacent sides already visible and one opposite corner missing. Five candidate markers A-E are placed near the missing corner; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the open broken line and the candidate markers, then draws two black closing segments of medium thickness from one visible endpoint to the correct candidate and from that candidate to the other visible endpoint so the full quadrilateral becomes a parallelogram. In the final state, only the correct missing-vertex marker changes to pale red fill with a dark red outline, while the other four candidates remain white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## perpendicular
- group: eyeballing
- ti2v:
  A square white canvas shows one black reference segment of medium thickness and one small hollow marker with white fill and a thick black outline centered on the marked through-point. Five candidate markers A-E sit near the hidden perpendicular direction on a short straight row; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the reference segment, the marked through-point marker, and the candidate row, then draws one black segment of medium thickness starting just outside that hollow marker and extending toward the correct candidate at a right angle to the reference segment. In the final state, only the candidate on this perpendicular segment changes to pale red fill with a dark red outline, while the other markers stay white. In portrait. Static camera.
- ti2i:
  A square white canvas shows one black reference segment of medium thickness and one small hollow marker with white fill and a thick black outline centered on the marked through-point. Five candidate markers A-E sit near the hidden perpendicular direction on a short straight row; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the reference segment, the marked through-point marker, and the candidate row, then draws one black segment of medium thickness starting just outside that hollow marker and extending toward the correct candidate at a right angle to the reference segment. In the final state, only the candidate on this perpendicular segment changes to pale red fill with a dark red outline, while the other markers stay white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## perpendicular_bisector
- group: eyeballing
- ti2v:
  A square white canvas shows two small hollow endpoint markers with white fill and thick black outlines connected by one black segment of medium thickness that stops at the marker boundaries. Five candidate markers A-E are arranged near the hidden perpendicular bisector; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the endpoint markers, the connecting segment, and the candidate markers, then draws a long black bisector line of medium thickness through the segment midpoint in the perpendicular direction so it runs across the canvas. In the final frame, only the candidate lying on that perpendicular bisector changes to pale red fill with a dark red outline, while all other markers remain white. In portrait. Static camera.
- ti2i:
  A square white canvas shows two small hollow endpoint markers with white fill and thick black outlines connected by one black segment of medium thickness that stops at the marker boundaries. Five candidate markers A-E are arranged near the hidden perpendicular bisector; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the endpoint markers, the connecting segment, and the candidate markers, then draws a long black bisector line of medium thickness through the segment midpoint in the perpendicular direction so it runs across the canvas. In the final frame, only the candidate lying on that perpendicular bisector changes to pale red fill with a dark red outline, while all other markers remain white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## polygon_sides_color
- group: visual_puzzle
- question:
  <QUESTION>
- options:
  [<OPTION_1>, <OPTION_2>, ...]
- answer:
  <OPTION_k>
- ti2v:
  Use the provided visual puzzle image as the starting frame. <QUESTION> A centered triangular arrangement on a clean white square canvas contains six filled regular polygons with thin black outlines, laid out as one on the top row, two on the middle row, and three on the bottom row. The polygons use a pastel palette, and every polygon with the same number of sides shares the same color even though triangles, quadrilaterals, pentagons, hexagons, heptagons, octagons, or nonagons may appear. One polygon is replaced by a light gray version with a black question mark, meaning its color is missing. The video holds on the puzzle frame, then smoothly crossfades to the solution so that only the question-mark region changes into the correct answer while every other shape stays fixed, and finally remains still on the solved image with a static camera and no zoom or pan.
- ti2i:
  Use the provided visual puzzle image as input. <QUESTION> Replace only the missing question-mark region with the correct answer while keeping every other shape unchanged. Output the solved image.
- ti2t:
  Use the provided visual puzzle image to solve the task. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. Answer with exactly one option.
- ti2t_answer:
  Answer: <OPTION_k>.
- ti2ti:
  Use the provided visual puzzle image as input. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. First determine the correct option, then replace only the missing question-mark region with that answer while keeping every other shape unchanged. Output the solved image.
- ti2ti_answer:
  {"text": "Answer: <OPTION_k>.", "image": "solution_image_path"}

## ray_intersection
- group: eyeballing
- ti2v:
  A square white canvas shows three thick dark gray line fragments placed near the outer part of the image so that only the edge-side portions of three rays are visible. Near the hidden common intersection, five candidate markers A-E are clustered as small white circles with thin dark gray outlines and black uppercase letters. The video first holds the three partial ray fragments and the candidate cluster, then extends each fragment inward with matching dark gray thickness until all three rays meet at one exact point. In the final state, only the candidate at that shared intersection changes to pale red fill with a dark red outline, while the other four markers stay white. In portrait. Static camera.
- ti2i:
  A square white canvas shows three thick dark gray line fragments placed near the outer part of the image so that only the edge-side portions of three rays are visible. Near the hidden common intersection, five candidate markers A-E are clustered as small white circles with thin dark gray outlines and black uppercase letters. The video first holds the three partial ray fragments and the candidate cluster, then extends each fragment inward with matching dark gray thickness until all three rays meet at one exact point. In the final state, only the candidate at that shared intersection changes to pale red fill with a dark red outline, while the other four markers stay white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## ray_reflect
- group: eyeballing
- ti2v:
  A square white canvas shows one black mirror segment of medium thickness, one small hollow source marker with white fill and a thick black outline away from the mirror, and only a short incoming black ray stub with a thin stroke extending from just outside the source marker toward the mirror. Five candidate markers A-E are placed near the hidden outgoing direction on a short row; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the mirror, the source marker, the short incoming stub, and the candidate markers, then extends that incoming ray to the reflection point on the mirror and continues it away from the mirror as a matching thin reflected segment toward the correct candidate. In the final frame, only the candidate on the reflected ray changes to pale red fill with a dark red outline, while the other markers remain white. In portrait. Static camera.
- ti2i:
  A square white canvas shows one black mirror segment of medium thickness, one small hollow source marker with white fill and a thick black outline away from the mirror, and only a short incoming black ray stub with a thin stroke extending from just outside the source marker toward the mirror. Five candidate markers A-E are placed near the hidden outgoing direction on a short row; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the mirror, the source marker, the short incoming stub, and the candidate markers, then extends that incoming ray to the reflection point on the mirror and continues it away from the mirror as a matching thin reflected segment toward the correct candidate. In the final frame, only the candidate on the reflected ray changes to pale red fill with a dark red outline, while the other markers remain white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## rectangle_height_color
- group: visual_puzzle
- question:
  <QUESTION>
- options:
  [<OPTION_1>, <OPTION_2>, ...]
- answer:
  <OPTION_k>
- ti2v:
  Use the provided visual puzzle image as the starting frame. <QUESTION> A centered row of seven tall rounded rectangles appears on a clean white square canvas. The bars are solid pastel colors with black outlines, equally spaced left to right, vertically centered, and they repeat three distinct heights so rectangles of the same height share the same color. The rightmost bar is shown as a light gray rounded rectangle with a black question mark, indicating the missing color for that height. The video holds on the puzzle frame, then smoothly crossfades to the solution so that only the question-mark region changes into the correct answer while every other shape stays fixed, and finally remains still on the solved image with a static camera and no zoom or pan.
- ti2i:
  Use the provided visual puzzle image as input. <QUESTION> Replace only the missing question-mark region with the correct answer while keeping every other shape unchanged. Output the solved image.
- ti2t:
  Use the provided visual puzzle image to solve the task. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. Answer with exactly one option.
- ti2t_answer:
  Answer: <OPTION_k>.
- ti2ti:
  Use the provided visual puzzle image as input. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. First determine the correct option, then replace only the missing question-mark region with that answer while keeping every other shape unchanged. Output the solved image.
- ti2ti_answer:
  {"text": "Answer: <OPTION_k>.", "image": "solution_image_path"}

## reflection
- group: eyeballing
- ti2v:
  A square white canvas shows one black reflection-axis segment of medium thickness and one small hollow source marker with white fill and a thick black outline on one side of that axis. Five candidate markers A-E are clustered near the hidden reflected location; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the reflection-axis segment, the source marker, and the candidate markers, then draws one black connector segment of medium thickness from just outside the source marker straight across the axis to the reflected candidate. In the final state, only the reflected-position candidate changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera.
- ti2i:
  A square white canvas shows one black reflection-axis segment of medium thickness and one small hollow source marker with white fill and a thick black outline on one side of that axis. Five candidate markers A-E are clustered near the hidden reflected location; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the reflection-axis segment, the source marker, and the candidate markers, then draws one black connector segment of medium thickness from just outside the source marker straight across the axis to the reflected candidate. In the final state, only the reflected-position candidate changes to pale red fill with a dark red outline, while the other four markers remain white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## right_triangle
- group: eyeballing
- ti2v:
  A square white canvas initially shows only five candidate markers A-E and no connecting geometry at all. Each candidate marker is a small white circle with a thin dark gray outline and a black uppercase letter, and exactly three of these five points form one right triangle. The video first holds the five markers alone, then draws a black triangle outline of medium thickness connecting the three special points in the order that closes one triangle. In the final state, only the marker at the 90 degree corner changes to pale red fill with a dark red outline, while the other four markers stay white. In portrait. Static camera.
- ti2i:
  A square white canvas initially shows only five candidate markers A-E and no connecting geometry at all. Each candidate marker is a small white circle with a thin dark gray outline and a black uppercase letter, and exactly three of these five points form one right triangle. The video first holds the five markers alone, then draws a black triangle outline of medium thickness connecting the three special points in the order that closes one triangle. In the final state, only the marker at the 90 degree corner changes to pale red fill with a dark red outline, while the other four markers stay white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## shape_reflect
- group: visual_puzzle
- question:
  <QUESTION>
- options:
  [<OPTION_1>, <OPTION_2>, ...]
- answer:
  <OPTION_k>
- ti2v:
  Use the provided visual puzzle image as the starting frame. <QUESTION> A clean white square canvas shows a horizontal black line across the middle dividing a top row and a bottom row of three pale green polygons. Matching shapes above and below the line are vertically mirrored versions of the same outlined triangle, square, pentagon, or hexagon. One position is replaced by a dotted outline circle with a black question mark, indicating the missing reflected shape that should appear there. The video holds on the puzzle frame, then smoothly crossfades to the solution so that only the question-mark region changes into the correct answer while every other shape stays fixed, and finally remains still on the solved image with a static camera and no zoom or pan.
- ti2i:
  Use the provided visual puzzle image as input. <QUESTION> Replace only the missing question-mark region with the correct answer while keeping every other shape unchanged. Output the solved image.
- ti2t:
  Use the provided visual puzzle image to solve the task. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. Answer with exactly one option.
- ti2t_answer:
  Answer: <OPTION_k>.
- ti2ti:
  Use the provided visual puzzle image as input. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. First determine the correct option, then replace only the missing question-mark region with that answer while keeping every other shape unchanged. Output the solved image.
- ti2ti_answer:
  {"text": "Answer: <OPTION_k>.", "image": "solution_image_path"}

## shape_size_grid
- group: visual_puzzle
- question:
  <QUESTION>
- options:
  [<OPTION_1>, <OPTION_2>, ...]
- answer:
  <OPTION_k>
- ti2v:
  Use the provided visual puzzle image as the starting frame. <QUESTION> A centered 3x3 arrangement on a clean white square canvas shows pale green outlined polygons with black borders. Across one axis the shape family stays constant and across the other axis the size steps from small to medium to large, using three of triangle, square, pentagon, and hexagon. One cell is empty except for a black question mark at its center, meaning the missing polygon of the correct shape and size must be inserted there. The video holds on the puzzle frame, then smoothly crossfades to the solution so that only the question-mark region changes into the correct answer while every other shape stays fixed, and finally remains still on the solved image with a static camera and no zoom or pan.
- ti2i:
  Use the provided visual puzzle image as input. <QUESTION> Replace only the missing question-mark region with the correct answer while keeping every other shape unchanged. Output the solved image.
- ti2t:
  Use the provided visual puzzle image to solve the task. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. Answer with exactly one option.
- ti2t_answer:
  Answer: <OPTION_k>.
- ti2ti:
  Use the provided visual puzzle image as input. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. First determine the correct option, then replace only the missing question-mark region with that answer while keeping every other shape unchanged. Output the solved image.
- ti2ti_answer:
  {"text": "Answer: <OPTION_k>.", "image": "solution_image_path"}

## size_cycle
- group: visual_puzzle
- question:
  <QUESTION>
- options:
  [<OPTION_1>, <OPTION_2>, ...]
- answer:
  <OPTION_k>
- ti2v:
  Use the provided visual puzzle image as the starting frame. <QUESTION> A clean white square canvas shows nine pale yellow circles with thin black outlines arranged in three spiral-like arms around the center. Along each arm the circles grow from small near the center to medium and large farther out, and the three arms are evenly spaced around the middle with a slight rotation. One circle position is replaced by a black question mark, indicating the missing circle size that belongs at that location. The video holds on the puzzle frame, then smoothly crossfades to the solution so that only the question-mark region changes into the correct answer while every other shape stays fixed, and finally remains still on the solved image with a static camera and no zoom or pan.
- ti2i:
  Use the provided visual puzzle image as input. <QUESTION> Replace only the missing question-mark region with the correct answer while keeping every other shape unchanged. Output the solved image.
- ti2t:
  Use the provided visual puzzle image to solve the task. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. Answer with exactly one option.
- ti2t_answer:
  Answer: <OPTION_k>.
- ti2ti:
  Use the provided visual puzzle image as input. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. First determine the correct option, then replace only the missing question-mark region with that answer while keeping every other shape unchanged. Output the solved image.
- ti2ti_answer:
  {"text": "Answer: <OPTION_k>.", "image": "solution_image_path"}

## size_grid
- group: visual_puzzle
- question:
  <QUESTION>
- options:
  [<OPTION_1>, <OPTION_2>, ...]
- answer:
  <OPTION_k>
- ti2v:
  Use the provided visual puzzle image as the starting frame. <QUESTION> A centered 3x3 grid of pale yellow circles with thin black outlines appears on a clean white square canvas. The four corner circles share one size, the four edge-middle circles share another size, and the center circle is a third size. One non-center position contains only a black question mark instead of a circle, indicating the missing circle size that must be filled in. The video holds on the puzzle frame, then smoothly crossfades to the solution so that only the question-mark region changes into the correct answer while every other shape stays fixed, and finally remains still on the solved image with a static camera and no zoom or pan.
- ti2i:
  Use the provided visual puzzle image as input. <QUESTION> Replace only the missing question-mark region with the correct answer while keeping every other shape unchanged. Output the solved image.
- ti2t:
  Use the provided visual puzzle image to solve the task. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. Answer with exactly one option.
- ti2t_answer:
  Answer: <OPTION_k>.
- ti2ti:
  Use the provided visual puzzle image as input. <QUESTION> Options: [<OPTION_1>, <OPTION_2>, ...]. First determine the correct option, then replace only the missing question-mark region with that answer while keeping every other shape unchanged. Output the solved image.
- ti2ti_answer:
  {"text": "Answer: <OPTION_k>.", "image": "solution_image_path"}

## square_outlier
- group: eyeballing
- ti2v:
  A square white canvas initially shows only five candidate markers A-E and no lines or helper shapes. Each candidate marker is a small white circle with a thin dark gray outline and a black uppercase letter, and exactly four of the five points are the vertices of one square while the fifth point is the outlier. The video first holds the five markers alone, then draws a black square outline of medium thickness through the four matching vertices, including rotation if the square is tilted. In the final state, only the single outlier marker changes to pale red fill with a dark red outline, while the four square-vertex markers remain white. In portrait. Static camera.
- ti2i:
  A square white canvas initially shows only five candidate markers A-E and no lines or helper shapes. Each candidate marker is a small white circle with a thin dark gray outline and a black uppercase letter, and exactly four of the five points are the vertices of one square while the fifth point is the outlier. The video first holds the five markers alone, then draws a black square outline of medium thickness through the four matching vertices, including rotation if the square is tilted. In the final state, only the single outlier marker changes to pale red fill with a dark red outline, while the four square-vertex markers remain white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

## triangle_center
- group: eyeballing
- ti2v:
  A square white canvas shows a triangle drawn only as a black outline of medium thickness, with no midpoint marks and no interior construction lines at first. Five candidate markers A-E are clustered near the hidden centroid; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the bare triangle and the five markers, then draws three black medians of medium thickness, each running from one triangle vertex to the midpoint of the opposite side so all three meet at one point. In the final state, only the centroid marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera.
- ti2i:
  A square white canvas shows a triangle drawn only as a black outline of medium thickness, with no midpoint marks and no interior construction lines at first. Five candidate markers A-E are clustered near the hidden centroid; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. The video first holds the bare triangle and the five markers, then draws three black medians of medium thickness, each running from one triangle vertex to the midpoint of the opposite side so all three meet at one point. In the final state, only the centroid marker changes to pale red fill with a dark red outline, while the other four markers remain white.
- ti2t:
  None
- ti2t_answer:
  Answer: <OPTION_LABEL>.
- ti2ti:
  None
- ti2ti_answer:
  {"text": "Answer: <OPTION_LABEL>.", "image": "solution_image_path"}

