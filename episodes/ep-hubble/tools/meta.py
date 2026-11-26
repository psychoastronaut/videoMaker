import json,os
R="episodes/ep-hubble/raw/"
img_side={
"hubble_primary_mirror_blank_being_ground_1979":"Large circular Hubble primary mirror blank on a grinding machine in a factory hall, a man in a white shirt and tie standing at the left, 1979, Perkin-Elmer optics fabrication",
"hubble_primary_mirror_polishing_swirl_pattern_1981":"Close view of the Hubble primary mirror surface being polished, white abrasive swirl pattern across the glass with a central hole, polishing machine arm above, two technicians in the background",
"engineers_inspecting_hubble_primary_mirror_blank_1981":"Two engineers examining the large honeycomb-backed Hubble primary mirror blank, a big white disc with a central hole on a rotating stand, 1981",
"technicians_inspecting_hubble_primary_mirror_in_clean_room_1982":"Technicians in white cleanroom suits and masks looking at the huge Hubble primary mirror standing upright in a blue-lit clean room, fisheye view, 1982",
"hubble_being_assembled_in_clean_room_1985":"Portrait photo of the Hubble Space Telescope body being assembled inside a tall clean room with scaffolding and a technician on a ladder, 1985",
"hubble_assembly_scaffolding_tower_1986":"Portrait photo of the Hubble Space Telescope flight article standing in a tall white assembly scaffolding tower with technicians far below, 1986",
"hubble_optical_telescope_assembly_labelled_cutaway":"Labelled NASA cutaway illustration of the Hubble Optical Telescope Assembly on a black background showing the 2.4 metre primary mirror, secondary mirror assembly, baffles, metering truss and instruments; the labels are tiny",
"hubble_lifted_into_workstand_at_kennedy_1989_small":"Small low-resolution photo: Hubble Space Telescope hanging vertically, lifted into a workstand at Kennedy Space Center, 1989, silver insulation and ring structure (deliberately the small 640 px rendition, to exercise upscaling)",
"hubble_deployed_from_discovery_black_space_1990":"Hubble Space Telescope just released from the shuttle Discovery on 25 April 1990: telescope with gold solar arrays and robotic arm against a completely black space background with a sun glint, part of the shuttle at bottom",
"astronauts_training_underwater_for_hubble_repair":"Astronauts in spacesuits with scuba divers training inside a huge underwater neutral buoyancy pool with a Hubble mock-up, blue water, STS-61 crew training, 1993",
"astronaut_hoffman_holding_wfpc1_during_hubble_repair":"Portrait photo of astronaut Jeffrey Hoffman in a white spacesuit on the shuttle robotic arm holding the old Wide Field/Planetary Camera (WF/PC-1) in space against black sky, December 1993",
"m100_galactic_nucleus_before_after_wfpc1_vs_wfpc2":"Side-by-side comparison print of the spiral galaxy M100 nucleus: left image blurry from the old Wide Field Planetary Camera 1, right image sharp from WFPC2 after the repair, title text M100 Galactic Nucleus on a white sheet",
"hourglass_nebula_wfpc2_poster":"Poster-style portrait image of the Hourglass Nebula MyCn18 in orange and yellow rings with a blue core, taken by Hubble WFPC2, with a caption box at the bottom",
"lagoon_nebula_wfpc2_poster_black_corner":"Poster-style portrait Hubble WFPC2 image of the Lagoon Nebula M8 with brown and blue gas twisters and a black stair-step corner cut out at the upper left (the WFPC2 chevron shape), caption box at the bottom",
"discovery_on_launch_pad_39b_march_1990_black_and_white":"Black and white photo of space shuttle Discovery on Launch Pad 39B across a marsh pond with grass in the foreground, rollout before the Hubble launch, March 1990",
"hoffman_musgrave_installing_wfpc2_artwork":"Illustration: two astronauts, Jeffrey Hoffman and Story Musgrave, installing the new Wide Field and Planetary Camera 2 on Hubble, blue Earth below, an open instrument bay and a dark camera box at the upper left",
"m100_through_three_cameras":"Wide strip showing three crops of the centre of spiral galaxy M100 side by side: a blurry pre-repair view, the WFPC2 view after repair and a later camera view; small labels under each",
"hubble_in_lockheed_clean_room_1988":"High-angle photo of the Hubble Space Telescope in the Lockheed clean room with a mechanical arm positioning an instrument and technicians in blue cleanroom suits around it, 1988",
"hubble_berthed_in_endeavour_payload_bay":"Hubble Space Telescope berthed upright in the shuttle Endeavour payload bay after capture, with brown solar arrays and blue Earth with clouds behind, December 1993",
"astronauts_akers_thornton_costar_installation_eva":"Astronaut Thomas Akers in a white spacesuit inside the open payload bay beside the Hubble aft shroud where COSTAR was installed, a robotic arm at left, 8 December 1993",
"sts61_servicing_mission_art_concept":"Artist concept of two astronauts servicing Hubble above blue Earth, opening the open instrument bay door with a dark camera module (STS-61 art, November 1993)",
"astronaut_thornton_lifts_costar_in_payload_bay":"Astronaut Kathryn Thornton lifting the large silver COSTAR box above the shuttle payload bay with Hubble's base and a second astronaut below, black background, 8 December 1993",
"hubble_fisheye_view_earth_australia":"Fisheye photo from the shuttle: Hubble Space Telescope tilted in the foreground against the curved Earth with white clouds and Australia, an astronaut's helmet at bottom, December 1993",
"hubble_approaches_endeavour_over_ocean":"Hubble Space Telescope seen as a small object approaching the shuttle Endeavour over the blue Indian Ocean and clouds, the shuttle robotic arm tip at lower right, 4 December 1993",
"new_solar_arrays_on_hubble_black_space":"Hubble Space Telescope with its new gold solar arrays deployed seen from the shuttle against black space with the curved Earth edge at the top, December 1993",
}
clip_side={
"hubble_in_orbit_flyby_animation":("NASA computer animation: camera flies past the Hubble Space Telescope in orbit, silver body and solar arrays, Earth limb with bright sun glare at the end","GSFC_20080520_HST_m10216_Whip_Pan","3-13 s of 22.6 s","NASA/GSFC"),
"hubble_closeup_over_clouds_animation":("NASA computer animation: slow close-up of the silver Hubble body and solar array with its aperture door opening, over a cloudy Earth","GSFC_20080519_HST_m10213-SlowLook1","0-12 s of 15 s","NASA/GSFC"),
"hubble_zoom_out_from_orbit_animation":("NASA computer animation: camera pulls away from Hubble over a dark cloudy Earth until it is a tiny dot, 720p","GSFC_20080520_HST_m10217_Zoom_Out","0-12 s of 20 s","NASA/GSFC"),
"shuttle_releases_hubble_over_clouds_animation":("NASA computer animation: space shuttle with open payload bay releasing Hubble, white clouds below, shuttle seen from above","GSFC_20080822_HST_m10235_Release_Fire_006","0-11 s of 14.4 s","NASA/GSFC"),
"shuttle_approaches_hubble_capture_animation":("NASA computer animation: space shuttle with Hubble berthed in its open payload bay over clouds, later wide view of Earth, from the servicing-mission rendezvous animation","GSFC_20080822_HST_m10329_Rendezvou_Capture","24-36 s of 52.7 s","NASA/GSFC"),
"astronaut_installs_instrument_on_hubble_animation":("NASA computer animation: spacewalking astronaut next to the open Hubble instrument bay with a robotic arm and a camera module being installed, Earth below","GSFC_20080822_HST_m10233_WFC3","10-22 s of 56.7 s","NASA/GSFC"),
"astronaut_opens_hubble_instrument_bay_animation":("NASA computer animation: astronaut at an open door of the Hubble instrument bay, Earth and clouds in the background, very steady","GSFC_20080822_HST_m10232_COS_Removal_Wide_C_007","0-10 s of 10.6 s","NASA/GSFC"),
"shuttle_atlantis_launch_with_caption":("Real footage of space shuttle Atlantis lifting off with a burned-in subtitle caption across the lower part of the frame (text baked into the video, STS-125 2009), from the NASA servicing mission 4 overview","GSFC_20190511_HST_m13186_SM4_Overview","2-11 s of 154 s","NASA/GSFC"),
"archival_hubble_on_shuttle_arm_over_earth":("Archival 1993-era footage: Hubble Space Telescope held by the shuttle robotic arm, close view of the dark telescope tube, then a wide view with solar arrays and sun glare, Earth below","GSFC_20190423_HST_m13176_TTSM1","38-50 s of 282 s","NASA/GSFC"),
"archival_spacewalk_closeup_hubble_repair":("Archival spacewalk footage during the Hubble repair: close view of Hubble hardware and handrails, then an astronaut in a white spacesuit working at the telescope; video quality is soft","GSFC_20190423_HST_m13176_TTSM1","86-98 s of 282 s","NASA/GSFC"),
"archival_spacewalk_astronauts_working_on_hubble":("Archival spacewalk footage: astronauts in white spacesuits working on Hubble hardware, handheld-looking and a bit unsteady; the last second or so may cut to a studio shot","GSFC_20190423_HST_m13176_TTSM1","168-180 s of 282 s","NASA/GSFC"),
"neutral_buoyancy_lab_astronaut_training_after_studio_intro":("Clip OPENS ON A STUDIO SHOT with two presenters and a white panel reading ASTRONAUT TRAINING VENUES for the first 9 seconds; the real content, astronauts training underwater in a neutral buoyancy pool, only starts after that. media_start: 9.2","GSFC_20190416_HST_m13175_TTAstronautTraining","68-82 s of 266 s","NASA/GSFC"),
"goddard_clean_room_technicians_synthetic_handheld_shake":("Technicians in white clean-room bunny suits around a spacecraft mock-up in a bright clean room. NOTE: a synthetic handheld shake was added to this clip on purpose (stabilization test); original footage steady","GSFC_20190416_HST_m13175_TTAstronautTraining","164-176 s of 266 s (+ synthetic shake)","NASA/GSFC"),
}
dia_side={
"perfect_mirror_all_rays_one_focus_diagram":"Diagram on dark blue: a concave mirror with parallel light rays all reflecting to one single focus point, labels 'ONE focus' and '= sharp star'",
"spherical_aberration_edge_rays_miss_focus_diagram":"Diagram on dark blue: concave mirror where edge rays (orange) cross the axis higher than centre rays (white), labels 'Edge rays focus higher', 'Centre rays focus lower', '= blurry star'; spherical aberration",
"mirror_shape_error_2p2_micrometres_diagram":"Diagram on dark blue: two mirror profile curves, intended shape in green and actual too-flat shape in red, label '2.2 µm at the edge', text 'A human hair is ~50x thicker'",
"null_corrector_field_lens_misplaced_1p3_mm_diagram":"Diagram on dark blue: Laser box, Null corrector box with a red misplaced field lens, Hubble mirror box; text 'field lens: 1.3 mm out of place', 'A worn spot on a measuring rod fooled the laser', 'the test says PERFECT for the wrong shape'",
"three_tests_two_dissent_trusted_one_wrong_diagram":"Diagram on dark blue: three rows, Null corrector says PERFECT (trusted), Test 2 says FLAW (dismissed), Test 3 says FLAW (dismissed); 'Lesson: investigate disagreement'",
"corrective_optics_cancel_mirror_error_diagram":"Diagram on dark blue: Main mirror (edge too flat) plus Corrector (opposite error) equals Sharp image (error cancelled); text 'WFPC2 and COSTAR, 1993'; glasses for a telescope",
"star_blur_halo_before_after_diagram":"Diagram on dark blue: one star drawn twice, left with a large fuzzy halo labelled 'Before: flawed mirror', right a tight sharp point labelled 'After: corrected'",
}
for k,v in img_side.items(): open(R+f"images/{k}.jpg.txt","w").write(v+"\n")
for k,v in clip_side.items(): open(R+f"clips/{k}.mp4.txt","w").write(v[0]+"\n")
for k,v in dia_side.items(): open(R+f"schematics/{k}.png.txt","w").write(v+"\n")
# SOURCES
meta={}
for l in open("scratch/img_meta.txt"):
    p=l.split("|"); meta[p[1]]=p
L=["# SOURCES — ep-hubble","","All media is NASA (US federal government work, public domain / NASA media usage guidelines: no copyright, credit NASA) or generated by us (CC0). Downloaded from the NASA Image and Video Library (images-api.nasa.gov).","","| file | URL | licence | author | note |","|---|---|---|---|---|"]
for k in img_side:
    m=meta[k+".jpg"]; L.append(f"| images/{k}.jpg | {m[2]} | Public domain (NASA) | {m[4]} | {m[3]}, {m[5]} |")
for k,v in clip_side.items():
    L.append(f"| clips/{k}.mp4 | https://images.nasa.gov/details/{v[1]} | Public domain (NASA) | {v[3]} | trimmed {v[2]}, re-encoded, audio removed |")
for k in dia_side: L.append(f"| schematics/{k}.png | generated (episodes/ep-hubble/tools/diag.py) | CC0 | project (Pillow) | original drawing |")
L+=["","PIA22913 credits NASA, ESA, STScI and image processing by Judy Schmidt (published by NASA/JPL as a PIA image).","Voice-over: piper en_US-ryan-high (local TTS)."]
open(R+"SOURCES.md","w").write("\n".join(L)+"\n")
# scene map
sm=[
("Nineteen ninety. NASA launches its flagship space telescope, and the first pictures come back blurry.",["deployed_from_discovery","launch_pad_39b","before_after_wfpc1"],["flyby","closeup","zoom_out","shuttle_releases","atlantis_launch","new_solar_arrays","fisheye","m100_through"]),
("Not a software bug. Not a dirty lens. The mirror itself was wrong.",["mirror_blank_being_ground","polishing_swirl","technicians_inspecting"],["inspecting_hubble_primary","mirror_in_clean_room","labelled_cutaway","mirror_shape_error"]),
("Meet Hubble. A bus-sized observatory circling about five hundred and forty kilometres above Earth.",["hubble_in_orbit_flyby","closeup_over_clouds","fisheye"],["zoom_out","new_solar_arrays","berthed_in_endeavour","approaches_endeavour","deployed_from","shuttle_releases","archival_hubble_on_shuttle_arm","labelled_cutaway"]),
("At its heart sits a two point four metre mirror, built to focus starlight into one sharp point.",["labelled_cutaway","mirror_blank_being_ground","inspecting_hubble_primary","perfect_mirror"],["mirror_in_clean_room","polishing_swirl","technicians_inspecting"]),
("NASA asked for a surface smooth to about ten nanometres. Engineers spent years grinding and polishing it.",["polishing_swirl","mirror_blank_being_ground","inspecting_hubble_primary"],["technicians_inspecting","mirror_in_clean_room"]),
("They nailed it. The mirror was flawlessly smooth. Just the wrong shape.",["mirror_in_clean_room","technicians_inspecting","mirror_shape_error"],["polishing_swirl","inspecting_hubble_primary","mirror_blank_being_ground"]),
("Near the edge, it was ground two point two micrometres too flat. A human hair is fifty times thicker.",["mirror_shape_error"],["perfect_mirror","spherical_aberration","polishing_swirl","mirror_blank_being_ground"]),
("Here is why that matters. A good mirror sends every ray, centre or edge, to the very same focus.",["perfect_mirror"],["spherical_aberration","mirror_shape_error"]),
("Hubble's mirror sends the edge rays to a different spot. That is called spherical aberration.",["spherical_aberration"],["perfect_mirror","mirror_shape_error"]),
("So every star becomes a bright point wrapped in a fuzzy halo, and most of the light lands where it should not.",["star_blur_halo","before_after_wfpc1"],["m100_through","spherical_aberration"]),
("How did nobody catch it? The mirror was checked with a test device called a null corrector.",["null_corrector"],["three_tests","lockheed_clean_room","assembled_in_clean_room","mirror_in_clean_room","technicians_inspecting","labelled_cutaway"]),
("The null corrector was assembled wrong. One lens sat one point three millimetres out of place.",["null_corrector"],["three_tests","lockheed_clean_room","mirror_shape_error"]),
("A measuring rod had a worn spot, the laser bounced off the wrong surface, and the instrument was built to match the mistake.",["null_corrector"],["three_tests","lockheed_clean_room","assembled_in_clean_room","assembly_scaffolding","clean_room"]),
("The test reported a perfect mirror. It was. For the wrong shape.",["three_tests","null_corrector"],["mirror_shape_error","perfect_mirror","technicians_inspecting","mirror_in_clean_room"]),
("Two other tests did show the flaw. They were dismissed as less accurate.",["three_tests"],["null_corrector","inspecting_hubble_primary","technicians_inspecting"]),
("Swapping a mirror in orbit was impossible. So engineers built glasses for a telescope.",["corrective_optics"],["flyby","zoom_out","closeup","fisheye","deployed_from","holding_wfpc1","costar_installation","new_solar_arrays"]),
("If you know the exact error, you can cancel it. Add mirrors with the opposite error, and the light comes out right.",["corrective_optics"],["star_blur","mirror_shape_error","perfect_mirror"]),
("The fix had two parts. A new camera with correction built in, and Costar, a box of small mirrors for the other instruments.",["holding_wfpc1","lifts_costar","costar_installation","installing_wfpc2_artwork"],["art_concept","instrument_on_hubble","opens_hubble_instrument","corrective_optics","labelled_cutaway","spacewalk"]),
("December nineteen ninety-three. Space Shuttle Endeavour reaches out and catches Hubble.",["approaches_endeavour","berthed_in_endeavour","shuttle_approaches_hubble_capture","archival_hubble_on_shuttle_arm"],["shuttle_releases","fisheye","atlantis_launch","art_concept","new_solar_arrays"]),
("Five spacewalks in eleven days. Astronauts swapped the hardware by hand, in orbit.",["archival_spacewalk","costar_installation","lifts_costar","installing_wfpc2_artwork","instrument_on_hubble"],["opens_hubble_instrument","holding_wfpc1","training_underwater","neutral_buoyancy","art_concept","fisheye"]),
("January nineteen ninety-four. The new view of galaxy M one hundred is razor sharp.",["before_after_wfpc1","m100_through"],["lagoon","hourglass","star_blur"]),
("Hubble went on to spend three decades rewriting astronomy.",["lagoon","hourglass"],["before_after","m100_through","flyby","zoom_out","closeup","fisheye","new_solar_arrays","deployed_from"]),
("The lesson. When three tests disagree, do not trust the one you like. Find out why.",["three_tests"],["null_corrector","star_blur","corrective_optics"]),
("A perfect result from a broken test is just a confident mistake.",["three_tests","null_corrector"],["mirror_shape_error","technicians_inspecting","perfect_mirror","mirror_in_clean_room"]),
]
js=[dict(scene=i+1,line=l,intended=a,acceptable=sorted(set(a+b))) for i,(l,a,b) in enumerate(sm)]
json.dump(js,open(R+"scene_map.json","w"),indent=1)
M=["# SCENE_MAP — ep-hubble (ground truth; id substring match)","","| # | narration | what the viewer should see | intended | acceptable |","|---|---|---|---|---|"]
for s in js: M.append(f"| {s['scene']} | {s['line']} | see intended | {', '.join(s['intended'])} | {', '.join(s['acceptable'])} |")
open(R+"SCENE_MAP.md","w").write("\n".join(M)+"\n")
print(len(js),"scenes")
