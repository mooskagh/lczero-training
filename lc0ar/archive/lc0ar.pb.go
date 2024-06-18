// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.30.0
// 	protoc        v4.25.3
// source: lc0ar.proto

package archive

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type TrainingDataArchive_PayloadType int32

const (
	TrainingDataArchive_PLAIN_DATA      TrainingDataArchive_PayloadType = 1
	TrainingDataArchive_TRANSPOSED_DATA TrainingDataArchive_PayloadType = 2
)

// Enum value maps for TrainingDataArchive_PayloadType.
var (
	TrainingDataArchive_PayloadType_name = map[int32]string{
		1: "PLAIN_DATA",
		2: "TRANSPOSED_DATA",
	}
	TrainingDataArchive_PayloadType_value = map[string]int32{
		"PLAIN_DATA":      1,
		"TRANSPOSED_DATA": 2,
	}
)

func (x TrainingDataArchive_PayloadType) Enum() *TrainingDataArchive_PayloadType {
	p := new(TrainingDataArchive_PayloadType)
	*p = x
	return p
}

func (x TrainingDataArchive_PayloadType) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (TrainingDataArchive_PayloadType) Descriptor() protoreflect.EnumDescriptor {
	return file_lc0ar_proto_enumTypes[0].Descriptor()
}

func (TrainingDataArchive_PayloadType) Type() protoreflect.EnumType {
	return &file_lc0ar_proto_enumTypes[0]
}

func (x TrainingDataArchive_PayloadType) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Do not use.
func (x *TrainingDataArchive_PayloadType) UnmarshalJSON(b []byte) error {
	num, err := protoimpl.X.UnmarshalJSONEnum(x.Descriptor(), b)
	if err != nil {
		return err
	}
	*x = TrainingDataArchive_PayloadType(num)
	return nil
}

// Deprecated: Use TrainingDataArchive_PayloadType.Descriptor instead.
func (TrainingDataArchive_PayloadType) EnumDescriptor() ([]byte, []int) {
	return file_lc0ar_proto_rawDescGZIP(), []int{1, 0}
}

type FileMetadata struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Name      *string `protobuf:"bytes,1,opt,name=name" json:"name,omitempty"`
	SizeBytes *uint32 `protobuf:"varint,2,opt,name=size_bytes,json=sizeBytes" json:"size_bytes,omitempty"`
}

func (x *FileMetadata) Reset() {
	*x = FileMetadata{}
	if protoimpl.UnsafeEnabled {
		mi := &file_lc0ar_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *FileMetadata) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*FileMetadata) ProtoMessage() {}

func (x *FileMetadata) ProtoReflect() protoreflect.Message {
	mi := &file_lc0ar_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use FileMetadata.ProtoReflect.Descriptor instead.
func (*FileMetadata) Descriptor() ([]byte, []int) {
	return file_lc0ar_proto_rawDescGZIP(), []int{0}
}

func (x *FileMetadata) GetName() string {
	if x != nil && x.Name != nil {
		return *x.Name
	}
	return ""
}

func (x *FileMetadata) GetSizeBytes() uint32 {
	if x != nil && x.SizeBytes != nil {
		return *x.SizeBytes
	}
	return 0
}

type TrainingDataArchive struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Magic                      *string                          `protobuf:"bytes,1,opt,name=magic" json:"magic,omitempty"`
	Version                    *uint32                          `protobuf:"varint,2,opt,name=version" json:"version,omitempty"`
	PayloadSizeBytes           *uint32                          `protobuf:"varint,3,opt,name=payload_size_bytes,json=payloadSizeBytes" json:"payload_size_bytes,omitempty"`
	PayloadType                *TrainingDataArchive_PayloadType `protobuf:"varint,4,opt,name=payload_type,json=payloadType,enum=archive.TrainingDataArchive_PayloadType" json:"payload_type,omitempty"`
	FileMetadata               []*FileMetadata                  `protobuf:"bytes,5,rep,name=file_metadata,json=fileMetadata" json:"file_metadata,omitempty"`
	TranspositionPageSizeBytes *uint32                          `protobuf:"varint,6,opt,name=transposition_page_size_bytes,json=transpositionPageSizeBytes" json:"transposition_page_size_bytes,omitempty"`
}

func (x *TrainingDataArchive) Reset() {
	*x = TrainingDataArchive{}
	if protoimpl.UnsafeEnabled {
		mi := &file_lc0ar_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *TrainingDataArchive) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*TrainingDataArchive) ProtoMessage() {}

func (x *TrainingDataArchive) ProtoReflect() protoreflect.Message {
	mi := &file_lc0ar_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use TrainingDataArchive.ProtoReflect.Descriptor instead.
func (*TrainingDataArchive) Descriptor() ([]byte, []int) {
	return file_lc0ar_proto_rawDescGZIP(), []int{1}
}

func (x *TrainingDataArchive) GetMagic() string {
	if x != nil && x.Magic != nil {
		return *x.Magic
	}
	return ""
}

func (x *TrainingDataArchive) GetVersion() uint32 {
	if x != nil && x.Version != nil {
		return *x.Version
	}
	return 0
}

func (x *TrainingDataArchive) GetPayloadSizeBytes() uint32 {
	if x != nil && x.PayloadSizeBytes != nil {
		return *x.PayloadSizeBytes
	}
	return 0
}

func (x *TrainingDataArchive) GetPayloadType() TrainingDataArchive_PayloadType {
	if x != nil && x.PayloadType != nil {
		return *x.PayloadType
	}
	return TrainingDataArchive_PLAIN_DATA
}

func (x *TrainingDataArchive) GetFileMetadata() []*FileMetadata {
	if x != nil {
		return x.FileMetadata
	}
	return nil
}

func (x *TrainingDataArchive) GetTranspositionPageSizeBytes() uint32 {
	if x != nil && x.TranspositionPageSizeBytes != nil {
		return *x.TranspositionPageSizeBytes
	}
	return 0
}

var File_lc0ar_proto protoreflect.FileDescriptor

var file_lc0ar_proto_rawDesc = []byte{
	0x0a, 0x0b, 0x6c, 0x63, 0x30, 0x61, 0x72, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x07, 0x61,
	0x72, 0x63, 0x68, 0x69, 0x76, 0x65, 0x22, 0x41, 0x0a, 0x0c, 0x46, 0x69, 0x6c, 0x65, 0x4d, 0x65,
	0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x12, 0x12, 0x0a, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x01,
	0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x12, 0x1d, 0x0a, 0x0a, 0x73, 0x69,
	0x7a, 0x65, 0x5f, 0x62, 0x79, 0x74, 0x65, 0x73, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x09,
	0x73, 0x69, 0x7a, 0x65, 0x42, 0x79, 0x74, 0x65, 0x73, 0x22, 0xf3, 0x02, 0x0a, 0x13, 0x54, 0x72,
	0x61, 0x69, 0x6e, 0x69, 0x6e, 0x67, 0x44, 0x61, 0x74, 0x61, 0x41, 0x72, 0x63, 0x68, 0x69, 0x76,
	0x65, 0x12, 0x14, 0x0a, 0x05, 0x6d, 0x61, 0x67, 0x69, 0x63, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09,
	0x52, 0x05, 0x6d, 0x61, 0x67, 0x69, 0x63, 0x12, 0x18, 0x0a, 0x07, 0x76, 0x65, 0x72, 0x73, 0x69,
	0x6f, 0x6e, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x07, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6f,
	0x6e, 0x12, 0x2c, 0x0a, 0x12, 0x70, 0x61, 0x79, 0x6c, 0x6f, 0x61, 0x64, 0x5f, 0x73, 0x69, 0x7a,
	0x65, 0x5f, 0x62, 0x79, 0x74, 0x65, 0x73, 0x18, 0x03, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x10, 0x70,
	0x61, 0x79, 0x6c, 0x6f, 0x61, 0x64, 0x53, 0x69, 0x7a, 0x65, 0x42, 0x79, 0x74, 0x65, 0x73, 0x12,
	0x4b, 0x0a, 0x0c, 0x70, 0x61, 0x79, 0x6c, 0x6f, 0x61, 0x64, 0x5f, 0x74, 0x79, 0x70, 0x65, 0x18,
	0x04, 0x20, 0x01, 0x28, 0x0e, 0x32, 0x28, 0x2e, 0x61, 0x72, 0x63, 0x68, 0x69, 0x76, 0x65, 0x2e,
	0x54, 0x72, 0x61, 0x69, 0x6e, 0x69, 0x6e, 0x67, 0x44, 0x61, 0x74, 0x61, 0x41, 0x72, 0x63, 0x68,
	0x69, 0x76, 0x65, 0x2e, 0x50, 0x61, 0x79, 0x6c, 0x6f, 0x61, 0x64, 0x54, 0x79, 0x70, 0x65, 0x52,
	0x0b, 0x70, 0x61, 0x79, 0x6c, 0x6f, 0x61, 0x64, 0x54, 0x79, 0x70, 0x65, 0x12, 0x3a, 0x0a, 0x0d,
	0x66, 0x69, 0x6c, 0x65, 0x5f, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x18, 0x05, 0x20,
	0x03, 0x28, 0x0b, 0x32, 0x15, 0x2e, 0x61, 0x72, 0x63, 0x68, 0x69, 0x76, 0x65, 0x2e, 0x46, 0x69,
	0x6c, 0x65, 0x4d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x52, 0x0c, 0x66, 0x69, 0x6c, 0x65,
	0x4d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x12, 0x41, 0x0a, 0x1d, 0x74, 0x72, 0x61, 0x6e,
	0x73, 0x70, 0x6f, 0x73, 0x69, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x70, 0x61, 0x67, 0x65, 0x5f, 0x73,
	0x69, 0x7a, 0x65, 0x5f, 0x62, 0x79, 0x74, 0x65, 0x73, 0x18, 0x06, 0x20, 0x01, 0x28, 0x0d, 0x52,
	0x1a, 0x74, 0x72, 0x61, 0x6e, 0x73, 0x70, 0x6f, 0x73, 0x69, 0x74, 0x69, 0x6f, 0x6e, 0x50, 0x61,
	0x67, 0x65, 0x53, 0x69, 0x7a, 0x65, 0x42, 0x79, 0x74, 0x65, 0x73, 0x22, 0x32, 0x0a, 0x0b, 0x50,
	0x61, 0x79, 0x6c, 0x6f, 0x61, 0x64, 0x54, 0x79, 0x70, 0x65, 0x12, 0x0e, 0x0a, 0x0a, 0x50, 0x4c,
	0x41, 0x49, 0x4e, 0x5f, 0x44, 0x41, 0x54, 0x41, 0x10, 0x01, 0x12, 0x13, 0x0a, 0x0f, 0x54, 0x52,
	0x41, 0x4e, 0x53, 0x50, 0x4f, 0x53, 0x45, 0x44, 0x5f, 0x44, 0x41, 0x54, 0x41, 0x10, 0x02, 0x42,
	0x29, 0x5a, 0x27, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x4c, 0x65,
	0x65, 0x6c, 0x61, 0x43, 0x68, 0x65, 0x73, 0x73, 0x5a, 0x65, 0x72, 0x6f, 0x2f, 0x6c, 0x63, 0x30,
	0x61, 0x72, 0x2f, 0x61, 0x72, 0x63, 0x68, 0x69, 0x76, 0x65,
}

var (
	file_lc0ar_proto_rawDescOnce sync.Once
	file_lc0ar_proto_rawDescData = file_lc0ar_proto_rawDesc
)

func file_lc0ar_proto_rawDescGZIP() []byte {
	file_lc0ar_proto_rawDescOnce.Do(func() {
		file_lc0ar_proto_rawDescData = protoimpl.X.CompressGZIP(file_lc0ar_proto_rawDescData)
	})
	return file_lc0ar_proto_rawDescData
}

var file_lc0ar_proto_enumTypes = make([]protoimpl.EnumInfo, 1)
var file_lc0ar_proto_msgTypes = make([]protoimpl.MessageInfo, 2)
var file_lc0ar_proto_goTypes = []interface{}{
	(TrainingDataArchive_PayloadType)(0), // 0: archive.TrainingDataArchive.PayloadType
	(*FileMetadata)(nil),                 // 1: archive.FileMetadata
	(*TrainingDataArchive)(nil),          // 2: archive.TrainingDataArchive
}
var file_lc0ar_proto_depIdxs = []int32{
	0, // 0: archive.TrainingDataArchive.payload_type:type_name -> archive.TrainingDataArchive.PayloadType
	1, // 1: archive.TrainingDataArchive.file_metadata:type_name -> archive.FileMetadata
	2, // [2:2] is the sub-list for method output_type
	2, // [2:2] is the sub-list for method input_type
	2, // [2:2] is the sub-list for extension type_name
	2, // [2:2] is the sub-list for extension extendee
	0, // [0:2] is the sub-list for field type_name
}

func init() { file_lc0ar_proto_init() }
func file_lc0ar_proto_init() {
	if File_lc0ar_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_lc0ar_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*FileMetadata); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_lc0ar_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*TrainingDataArchive); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_lc0ar_proto_rawDesc,
			NumEnums:      1,
			NumMessages:   2,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_lc0ar_proto_goTypes,
		DependencyIndexes: file_lc0ar_proto_depIdxs,
		EnumInfos:         file_lc0ar_proto_enumTypes,
		MessageInfos:      file_lc0ar_proto_msgTypes,
	}.Build()
	File_lc0ar_proto = out.File
	file_lc0ar_proto_rawDesc = nil
	file_lc0ar_proto_goTypes = nil
	file_lc0ar_proto_depIdxs = nil
}