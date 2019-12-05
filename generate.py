from keramuse.AnnMidiProcessor import AnnMidiProcessor
from parse_args import parse_args

args = parse_args()
amp = AnnMidiProcessor(track=args.track, sequence_length=args.sequence_length)
amp.load_model(loss=args.loss, opt=args.opt)

if args.midis > 1:
    for i in range(args.midis):
        amp.generate_midi(notes_nb=args.notes, destination='{}{}'.format(args.destination, i + 1),
                          instrumentName=args.instrument, bars=args.bars)
else:
    amp.generate_midi(notes_nb=args.notes, destination=args.destination, instrumentName=args.instrument,
                      bars=args.bars)
