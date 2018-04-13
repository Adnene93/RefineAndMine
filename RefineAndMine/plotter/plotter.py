
from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pylab
from bisect import bisect_left
import matplotlib.backends.backend_pdf as backend_pdf
from PyPDF2 import PdfFileReader, PdfFileWriter



def pdf_cat(input_files, destination_file):
    input_streams = []
    try:
        # First open all the files, then produce the output file, and
        # finally close the input files. This is necessary because
        # the data isn't read from the input files until the write
        # operation. Thanks to
        # https://stackoverflow.com/questions/6773631/problem-with-closing-python-pypdf-writing-getting-a-valueerror-i-o-operation/6773733#6773733
        for input_file in input_files:
            input_streams.append(open(input_file, 'rb'))
        writer = PdfFileWriter()
        for reader in map(PdfFileReader, input_streams):
            for n in range(reader.getNumPages()):
                writer.addPage(reader.getPage(n))
        
        with open(destination_file, 'wb') as fout:
        	writer.write(fout)
        #writer.write(output_stream)
    finally:
        for f in input_streams:
            f.close()



rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
# FONTSIZE = 15
# MARKERSIZE = 2
# LINEWIDTH = 2
#332288, #88CCEE, #44AA99, #117733, #999933, #DDCC77, #CC6677, #882255, #AA4499, #117733

curvesname = {"quality":"Quality","guarantee":"QualityBound","crispiness":"SpecificityBound","recall":"Diversity","globality":"Specificity",
			  "quality_RefineAndMine":"R\&M\_Quality","guarantee_RefineAndMine":"R\&M\_Guarantee","recall_RefineAndMine":"R\&M\_Diversity",
			  "quality_MCTS4DM":"MCTS\_Quality","guarantee_MCTS4DM":"MCTS\_Guarantee","recall_MCTS4DM":"MCTS\_Diversity",
			   }
markerByOpt = {"quality":"","guarantee":""}
lineTypeByOpt = {"quality":'-',"guarantee":'-'}
methodColors={"quality":'#27ae60',"guarantee":'#332288',"globality":'#e67e22',"recall":'#9b59b6','crispiness':'#e74c3c',
			  "quality_RefineAndMine":'#27ae60',"guarantee_RefineAndMine":'#332288',"recall_RefineAndMine":'#9b59b6','crispiness_RefineAndMine':'#e74c3c',"globality_RefineAndMine":'#e67e22',
			  "quality_MCTS4DM":'#55efc4',"recall_MCTS4DM":'#fd79a8'}
#'_MCTS4DM'
#'_RefineAndMine'


alphas_prevalence={
	
	'HABERMAN_2':0.2647058823529412,
	'GLASS':0.32710280373831774,
	'CREDITA_+':0.44894894894894893,
	'ABALONE_M':0.3658127842949485,

}

REARRANGE=False
CONFIGURATION="Q1"
if CONFIGURATION=="Q1": #QUALITY AND GUARANTEE
	FONTSIZE = 38
	LEGENDFONTSIZE = 32
	MARKERSIZE = 4
	LINEWIDTH = 4
	FIGSIZE=(11.5, 3.5)
	LOG_SCALE_X=False
	#XAXISVALUE_REDUCEPRECISION=False
	X_SCALE_BARS="Time (s)"
	Y_SCALE_BARS="Quality"
	TIME_NORMALIZE=False
	TITLE=""
	LEGENDSHOW=False
	SHOWEXHAUSTIVELINE=True
	SHOWConfirmationLINE=True
	SHOWQualityFoundLINE=True


if CONFIGURATION=="Q2": #QUALITY AND GUARANTEE
	FONTSIZE = 38
	LEGENDFONTSIZE = 32
	MARKERSIZE = 4
	LINEWIDTH = 4
	#FIGSIZE=(11.5, 4)
	FIGSIZE=(11.5, 3.5)
	LOG_SCALE_X=False
	#XAXISVALUE_REDUCEPRECISION=False
	X_SCALE_BARS="Time (s)"
	Y_SCALE_BARS="Spec/Div"
	TIME_NORMALIZE=False
	TITLE=""
	LEGENDSHOW=False
	SHOWEXHAUSTIVELINE=True
	SHOWConfirmationLINE=True
	SHOWQualityFoundLINE=True

if CONFIGURATION=="Q3": #QUALITY AND GUARANTEE
	FONTSIZE = 38
	LEGENDFONTSIZE = 32
	MARKERSIZE = 4
	LINEWIDTH = 4
	FIGSIZE=(11.5, 4)
	LOG_SCALE_X=True
	#XAXISVALUE_REDUCEPRECISION=False
	
	Y_SCALE_BARS="Metrics"
	TIME_NORMALIZE=False
	TITLE=""
	LEGENDSHOW=False
	SHOWEXHAUSTIVELINE=False
	SHOWConfirmationLINE=True
	SHOWQualityFoundLINE=True
	if LOG_SCALE_X:
		X_SCALE_BARS="Time (s)"
	else:
		X_SCALE_BARS="Time (s)"
	REARRANGE=True

if CONFIGURATION=="LEGENDQ1": #QUALITY AND GUARANTEE
	FONTSIZE = 38
	LEGENDFONTSIZE = 32
	MARKERSIZE = 4
	LINEWIDTH = 4
	FIGSIZE=(18, 4)
	LOG_SCALE_X=False
	#XAXISVALUE_REDUCEPRECISION=False
	X_SCALE_BARS="Time (s)"
	Y_SCALE_BARS="Quality"
	TIME_NORMALIZE=False
	TITLE=""
	LEGENDSHOW=True
	figsizeLegend=(23.1,0.85)
	bbox_to_anchorLegend=(-0.007, 1.2)
	nb_col_legend=5
	filename="LegendQ1.pdf"
	SHOWEXHAUSTIVELINE=True
	SHOWConfirmationLINE=True
	SHOWQualityFoundLINE=True




if CONFIGURATION=="LEGENDQ2": #QUALITY AND GUARANTEE
	FONTSIZE = 38
	LEGENDFONTSIZE = 32
	MARKERSIZE = 4
	LINEWIDTH = 4
	FIGSIZE=(18, 4)
	LOG_SCALE_X=False
	#XAXISVALUE_REDUCEPRECISION=False
	X_SCALE_BARS="Time (s)"
	Y_SCALE_BARS="Recall"
	TIME_NORMALIZE=False
	TITLE=""
	LEGENDSHOW=True
	figsizeLegend=(17.39,0.85)
	bbox_to_anchorLegend=(-0.01, 1.2)
	nb_col_legend=4
	filename="LegendQ2.pdf"
	SHOWEXHAUSTIVELINE=True
	SHOWConfirmationLINE=True
	SHOWQualityFoundLINE=True


if CONFIGURATION=="LEGENDQ3": #QUALITY AND GUARANTEE
	FONTSIZE = 38
	LEGENDFONTSIZE = 32
	MARKERSIZE = 4
	LINEWIDTH = 4
	FIGSIZE=(18, 4)
	LOG_SCALE_X=False
	#XAXISVALUE_REDUCEPRECISION=False
	X_SCALE_BARS="Time (s)"
	Y_SCALE_BARS="Quality"
	TIME_NORMALIZE=False
	TITLE=""
	LEGENDSHOW=True
	figsizeLegend=(24.1,0.85)
	bbox_to_anchorLegend=(-0.007, 1.2)
	nb_col_legend=10
	filename="LegendQ3.pdf"
	SHOWEXHAUSTIVELINE=False
	SHOWConfirmationLINE=False
	SHOWQualityFoundLINE=False
	REARRANGE=True


def plot_timeseries(timeseries,xlogscale=LOG_SCALE_X,ylogscale=False,destination_file=None,exhaustive_time=None,confirmation_time=None,best_pattern_found_time=None,show_plot=True): #timeseries={'timeserieName':(x_axis_values,y_axis_values) or y_axis_values }
	
	
	fig,ax = plt.subplots(figsize=FIGSIZE)
	# ax.set_xlim([-5,205])
	TIMELIMIT=7200
	colors=['#332288', '#27ae60','#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499', '#117733','#e67e22','#9b59b6']
	ax.set_ylim([-0.02,1.02])
	# if CONFIGURATION=="Q3":
	# 	ax.set_xlim([10**-3,10**3])
	ind_color=0
	for method in  sorted(timeseries) :
		vector=timeseries[method]
		#print method
		if method=='globality' and False:
			vector=(vector[0],[1-x for x in vector[1]])
		if type(vector) is tuple:
			iterations=vector[0]	
			vector=vector[1]
			if vector[0]<0:
				continue


			if REARRANGE and ('quality' in method or 'guarantee' in method): #WRACC -> Informedness
				consider_alpha_of='XXX'
				if 'HABER' in destination_file:
					consider_alpha_of='HABERMAN_2'
				if 'GLA' in destination_file:
					consider_alpha_of='GLASS'
				if 'CRED' in destination_file:
					consider_alpha_of='CREDITA_+'
				if 'ABAL' in destination_file:
					consider_alpha_of='ABALONE_M'
				
				
				
				vector=[x*(1./(alphas_prevalence[consider_alpha_of]*(1-alphas_prevalence[consider_alpha_of]))) for x in vector]

			index_7200=bisect_left(iterations,TIMELIMIT)
			iterations=iterations[:index_7200]
			vector=vector[:index_7200]
		else:
			iterations = range(1,len(vector)+1)

		plt.plot(iterations, vector, label=curvesname.get(method,method), linewidth=LINEWIDTH,color=methodColors.get(method,colors[ind_color%len(colors)]),linestyle=lineTypeByOpt.get(method,'-'), marker=markerByOpt.get(method,''))
		ind_color+=1
	if xlogscale : ax.set_xscale("log")
	if ylogscale : ax.set_yscale("log")
	ax.set_xlabel(X_SCALE_BARS + ' - ' + destination_file.replace('_','\_'),fontsize=FONTSIZE)
	ax.set_ylabel(Y_SCALE_BARS,fontsize=FONTSIZE)
	ax.tick_params(axis='x', labelsize=FONTSIZE)
	ax.tick_params(axis='y', labelsize=FONTSIZE)
	if exhaustive_time is not None:
		if exhaustive_time<TIMELIMIT and SHOWEXHAUSTIVELINE:
			ax.axvline(exhaustive_time, 0, 1.03,color="#e74c3c",linewidth=LINEWIDTH-1,linestyle="--",markersize=MARKERSIZE-1,alpha=0.85,label='ExhaustiveTime')#)
	if confirmation_time is not None:
		if SHOWQualityFoundLINE:
			ax.axvline(confirmation_time, 0, 1.03,color="#9b59b6",linewidth=LINEWIDTH-1,linestyle="--",markersize=MARKERSIZE-1,alpha=0.85,label='ConfirmationTime')#,)
		else:
			ax.axvline(confirmation_time, 0, 1.03,color="#9b59b6",linewidth=LINEWIDTH-1,linestyle="--",markersize=MARKERSIZE-1,alpha=0.85)

	if best_pattern_found_time is not None:
		if SHOWQualityFoundLINE:
			ax.axvline(best_pattern_found_time, 0, 1.03,color="#16a085",linewidth=LINEWIDTH-1,linestyle="--",markersize=MARKERSIZE-1,alpha=0.85,label='BestFoundTime')#,)
		else:
			ax.axvline(best_pattern_found_time, 0, 1.03,color="#16a085",linewidth=LINEWIDTH-1,linestyle="--",markersize=MARKERSIZE-1,alpha=0.85)#,)


	if True:
		plt.yticks(np.array([0.,0.25,0.5,0.75,1.]))
		plt.setp(ax.get_xticklabels()[0], visible=False)    
    	plt.setp(ax.get_xticklabels()[-1], visible=False)

	#legend = ax.legend(loc='upper right', shadow=True, fontsize=FONTSIZE)
	if LEGENDSHOW:
		legend=ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.07),ncol=3,fancybox=True,fontsize=LEGENDFONTSIZE, framealpha=1.)#framealpha=0.85
		#figlegend.legend(lines, ('one', 'two'), 'center')
		figlegend = pylab.figure(figsize=figsizeLegend)
		pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left',bbox_to_anchor=bbox_to_anchorLegend,ncol=nb_col_legend,fancybox=True,fontsize=LEGENDFONTSIZE, framealpha=1.)
		figlegend.savefig(filename)

	#plt.title(destination_file.replace('_','\_'), fontsize=FONTSIZE, fontweight='bold')
	
	#fig.suptitle()
	#plt.show()
	if destination_file is not None:
		fig.tight_layout()
		plt.savefig('./'+destination_file+'.pdf')
	if show_plot:
		plt.show()














LIST_1=[ [[30.0, 42.0], [58.0, 65.0]]    ,  [[47.0, 76.0], [66.0, 69.0]]    ]
LIST_2=[ [[30.0, 42.0], [58.0, 65.0]]    ,  [[48.0, 64.0], [60.0, 64.0]]    ]
def plot_patterns(dataset,attributes,positive_extent,negative_extent,exhaustive_patterns=[],algo_discretization=iter([LIST_1,LIST_2]), show = True, save_in = None,title_for_figure='HABERMAN\_2\_1',timespent_by_exhaustive=0.5):
		
		curve_also=True
		SHOW_DISCRETIZATIONS_LINE=False
		# print timespent_by_exhaustive*4
		# raw_input('....')



		positive_array=[]
		negative_array=[]
		for i,row in enumerate(dataset):
			if row['positive']:
				positive_array.append([row[a] for a in attributes])
			else:
				negative_array.append([row[a] for a in attributes])
		
		

		positive_array=np.array(positive_array)
		negative_array=np.array(negative_array)


		figure, ax = plt.subplots()#plt.subplots(1,2)
		if curve_also:
			ax_0=plt.subplot(221)
			ax_1=plt.subplot(222)
			ax_2=plt.subplot(212)
		else:
			ax_0=plt.subplot(121)
			ax_1=plt.subplot(122)


		xminlim=min(min(positive_array[:,0]),min(negative_array[:,0]))
		xmaxlim=max(max(positive_array[:,0]),max(negative_array[:,0]))

		yminlim=min(min(positive_array[:,1]),min(negative_array[:,1]))
		ymaxlim=max(max(positive_array[:,1]),max(negative_array[:,1]))

		ax_0.set_xlim(xminlim*(0.99),xmaxlim*(1.01))
		ax_0.set_ylim(yminlim*(0.99),ymaxlim*(1.01))

		ax_1.set_xlim(xminlim*(0.99),xmaxlim*(1.01))
		ax_1.set_ylim(yminlim*(0.99),ymaxlim*(1.01))
		# ax_1.set_xlim(min(positive_array[:,0]),max(positive_array[:,0]))
		# ax_1.set_ylim(min(positive_array[:,1]),max(positive_array[:,1]))
		

		# print min(positive_array[:,1]),max(positive_array[:,1])
		# print min(negative_array[:,0]),max(negative_array[:,0])
		# print min(negative_array[:,1]),max(negative_array[:,1])
		#
		# ax_2.set_ylim([-0.01,1.01])
		# #ax_2.set_xlim([-1,60])
		# #ax_2.set_xlim([-0.01,2.1])
		# ax_2.set_xlim([-0.01,timespent_by_exhaustive*4])#timespent_by_exhaustive
		#

		if len(positive_array)>0 :
			ax_0.scatter(positive_array[:,0],positive_array[:,1],color="green",marker="$+$")
			ax_1.scatter(positive_array[:,0],positive_array[:,1],color="green",marker="$+$")
		if len(negative_array)>0 :
			ax_0.scatter(negative_array[:,0],negative_array[:,1],color="red",marker="$-$")
			ax_1.scatter(negative_array[:,0],negative_array[:,1],color="red",marker="$-$")

		for pattern in exhaustive_patterns:
			#pattern=[[30.0, 64.0], [60.0, 64.0]]
			pattern=np.array(pattern)
			#print pattern[0,0]
			rectas_start=tuple([pattern[i,0] for i,a in enumerate(attributes)])
			rectas_differences=[pattern[i,1]-pattern[i,0] for i,a in enumerate(attributes)]

			rect = patches.Rectangle(rectas_start,rectas_differences[0],rectas_differences[1],linewidth=2,facecolor='none',alpha=0.4,color='green',edgecolor='blue')

			# Add the patch to the Axes
			ax_0.add_patch(rect)

		def init():
			return 
		

		
		def update_plot(patterns_list,cutty_points):
			ax_1.clear()
			ax_1.set_xlim(xminlim*(0.99),xmaxlim*(1.01))
			ax_1.set_ylim(yminlim*(0.99),ymaxlim*(1.01))
			ax_1.scatter(positive_array[:,0],positive_array[:,1],color="green",marker="$+$")
			ax_1.scatter(negative_array[:,0],negative_array[:,1],color="red",marker="$-$")
			if SHOW_DISCRETIZATIONS_LINE:
				for i,attr in enumerate(attributes):
					cutty_points_attr=cutty_points[i]
					#print cutty_points_attr
					#print '---------------'
					if i==0:

						for ic in range(0,len(cutty_points_attr),2):
							c=cutty_points_attr[ic]

							if ic==0:
								c=c-1
							elif ic==len(cutty_points_attr)-1:
								c=c+1
							else:
								c=(c+cutty_points_attr[ic-1])/2.
								ic+=1
							ax_1.axvline(c, 0, 1000,color="gray",linewidth=1,linestyle=":",markersize=10,alpha=1)
						ax_1.axvline(cutty_points_attr[-1]+1, 0, 1000,color="gray",linewidth=1,linestyle=":",markersize=10,alpha=1)
					else:

						for ic in range(0,len(cutty_points_attr),2):
							c=cutty_points_attr[ic]
							if ic==0:
								c=c-1
							elif ic==len(cutty_points_attr)-1:
								c=c+1
							else:
								c=(c+cutty_points_attr[ic-1])/2.
								ic+=1
							ax_1.axhline(c, 0, 1000,color="gray",linewidth=1,linestyle=":",markersize=10,alpha=1)
						ax_1.axhline(cutty_points_attr[-1]+1, 0, 1000,color="gray",linewidth=1,linestyle=":",markersize=10,alpha=1)
			
			for pattern in patterns_list:
				pattern=np.array(pattern)
				rectas_start=tuple([pattern[i,0] for i,a in enumerate(attributes)])
				rectas_differences=[pattern[i,1]-pattern[i,0] for i,a in enumerate(attributes)]
				rect = patches.Rectangle(rectas_start,rectas_differences[0],rectas_differences[1],linewidth=2,facecolor='none',alpha=0.4,color='green',edgecolor='blue')
				ax_1.add_patch(rect)


		def update_curve(timeseries):
			ax_2.clear()
			colors=['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499', '#117733']
			ax_2.set_ylim([-0.01,1.01])
			#ax_2.set_xlim([-1,60])
			#ax_2.set_xlim([-0.01,2.1])
			ax_2.set_xlim([-0.01,timespent_by_exhaustive*4])#timespent_by_exhaustive

			ind_color=3
			for method in  sorted(timeseries) :
				vector=timeseries[method]
				if type(vector) is tuple:
					iterations=vector[0]
					vector=vector[1]
				else:
					iterations = range(1,len(vector)+1)

				#ax_2.plot(iterations, vector, label=method, linewidth=1,color=colors[ind_color%len(colors)])
				ax_2.plot(iterations, vector, label=curvesname.get(method,method), linewidth=LINEWIDTH,color=methodColors.get(method,colors[ind_color%len(colors)]),linestyle=lineTypeByOpt.get(method,'-'), marker=markerByOpt.get(method,''))
				#ax_2.legend_.remove()
				#legend = ax_2.legend(loc='upper right', shadow=True, fontsize=FONTSIZE)
				legend=ax_2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),ncol=5,fancybox=True,fontsize=LEGENDFONTSIZE-24, framealpha=.85)#framealpha=0.85
				ind_color+=1

		TEMPS=[0]
		
		#pdf = backend_pdf.PdfPages("./tmp/merged.pdf")
		pdfFiles=[]
		DONE=True

		def animate(ev=True):
			

			try:
				patterns_list,cutty_points,timeseries,temps=next(algo_discretization)
				TEMPS.append(temps[-1])
				#ani.event_source.interval = TEMPS[-1]-TEMPS[-2]
				get_next.SEEN_ALREADY.append((patterns_list,cutty_points,timeseries))
				update_plot(patterns_list,cutty_points)
				if curve_also:
					#print len(timeseries)
					ax_2=plt.subplot(212)
					update_curve(timeseries)
					
					#ax_2.axvline(get_next.STEP, 0, 1000,color="red",linewidth=1,linestyle=":",markersize=10,alpha=1)
					
					ax_2.axvline(temps[-1], 0, 1000,color="red",linewidth=1,linestyle=":",markersize=10,alpha=1)
				
				plt.title(title_for_figure.replace('_','\_')+' - STEP ' + str(animate.STEP))
				
				ax_0.title.set_text('Exhaustive Search Results')
				ax_1.title.set_text('Refine and Mine Results')
				
				animate.STEP=animate.STEP+1
				get_next.STEP=animate.STEP
				#pdf.savefig(figure)
				plt.savefig('./tmp/F'+str(animate.STEP).zfill(3)+'.pdf')
				pdfFiles.append('./tmp/F'+str(animate.STEP).zfill(3)+'.pdf')


			except Exception as e:
				DONE=True
				plt.title(title_for_figure.replace('_','\_')+' - ENDED')

		
		
		def get_next(ev):
			

			try:

				if ev.key=='right':
					if get_next.STEP>=len(get_next.SEEN_ALREADY):
						patterns_list,cutty_points,timeseries,temps=next(algo_discretization)
						get_next.SEEN_ALREADY.append((patterns_list,cutty_points,timeseries))
						update_plot(patterns_list,cutty_points)
						get_next.STEP=get_next.STEP+1
						if curve_also:
							update_curve(get_next.SEEN_ALREADY[-1][2])
							ax_2.axvline(get_next.STEP, 0, 1000,color="red",linewidth=1,linestyle=":",markersize=10,alpha=1)
						plt.title('STEP ' + str(get_next.STEP))
						
					else:
						
						patterns_list,cutty_points,timeseries=get_next.SEEN_ALREADY[get_next.STEP]
						get_next.STEP=get_next.STEP+1
						update_plot(patterns_list,cutty_points)
						if curve_also:
							update_curve(get_next.SEEN_ALREADY[-1][2])
							ax_2.axvline(get_next.STEP, 0, 1000,color="red",linewidth=1,linestyle=":",markersize=10,alpha=1)
						plt.title('STEP ' + str(get_next.STEP))
						


				elif ev.key=='left':
					if get_next.STEP>=2:
						get_next.STEP=get_next.STEP-1
						patterns_list,cutty_points,timeseries=get_next.SEEN_ALREADY[get_next.STEP-1]
						update_plot(patterns_list,cutty_points)
						if curve_also:
							update_curve(get_next.SEEN_ALREADY[-1][2])
							ax_2.axvline(get_next.STEP, 0, 1000,color="red",linewidth=1,linestyle=":",markersize=10,alpha=1)
						plt.title('STEP ' + str(get_next.STEP))
						
				

			except Exception as e:
				print e
				plt.title('END !!')

		animate.STEP=1
		get_next.STEP=0
		get_next.SEEN_ALREADY=[]
		init()


		def on_keyboard(event):
			ani.event_source.stop()
			if event.key == 'right':
				get_next(event)
				plt.draw()
			elif event.key == 'left':
				get_next(event)
				plt.draw()

			#plt.clf()
			#plt.plot(data**power)
			#plt.draw()


		from matplotlib import animation
		plt.gcf().canvas.mpl_connect('key_pwress_event', on_keyboard)
		ani = animation.FuncAnimation(figure, animate, init_func = init, interval=0.001)
		if False:
			
			Writer = animation.writers['ffmpeg']
			writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=5000)
			ani.save('./tmp/'+title_for_figure+'.mp4', writer=writer,dpi=750)
		
		else:
			while not DONE:
				animate()

		#ani = animation.FuncAnimation(figure, animate, init_func = init, frames=50)
		#ani.save('HABERMAN_2_1.gif', fps=30, writer='imagemagick')
		#plt.show()
		
		pdfFiles=['./tmp/F'+str(x).zfill(3)+'.pdf' for x in range(2,139)]
		pdf_cat(pdfFiles, './tmp/'+title_for_figure+'.pdf')




		#raw_input('********************************')

		# figure, ax = plt.subplots()
		# points = np.array([p.tofloat() for p in self.data.points_2])
		# positive_points = points[self.data.positive_instance_indices] if len(self.data.positive_instance_indices) > 0 else np.array([])
		# negative_points = points[self.data.negative_instance_indices] if len(self.data.negative_instance_indices) > 0 else np.array([])

		# ax.set_xlabel(self.data.column_1_name)
		# ax.set_ylabel(self.data.column_2_name)

		# if len(positive_points)>0 :
		#     ax.scatter(positive_points[:,0],positive_points[:,1],c="green",marker="$+$")
		# if len(negative_points)>0 :
		#     ax.scatter(negative_points[:,0],negative_points[:,1],c="red",marker="$-$")

		# all_extent = set()
		# for pattern in patterns :
		#     all_extent |= pattern.extent

		# positive_extent = self.positive_instances_in_extent(all_extent)
		# negative_extent = self.negative_instances_in_extent(all_extent)

		# if len(positive_extent)>0 :
		#     ax.scatter(points[positive_extent,0],points[positive_extent,1],c="blue",marker="$+$")
		# if len(negative_extent)>0 :
		#     ax.scatter(points[negative_extent,0],points[negative_extent,1],c="blue",marker="$-$")
		# for pattern in patterns:
		#     pattern.plot(ax)


		# wracc = self.wracc(all_extent)
		# best_wracc = self.wracc_measure.best(self.wracc_measure.positives_prevalence)
		# percentage_to_best = 100*(wracc/best_wracc)
		# plt.title("support = "+str(len(all_extent))+", number of patterns = "+str(len(patterns))+", wracc = "+('%.2f'%wracc)+" ("+('%.2f'%percentage_to_best)+" %"+" from best)")
		#figure.tight_layout()

		# if save_in != None:
		#     plt.savefig(save_in)
		# if show :
		#     plt.show()